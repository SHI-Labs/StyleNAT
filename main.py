import hydra
import os
import random
import warnings
from datetime import timedelta
import logging
import torch
from omegaconf import OmegaConf, open_dict

from models.generator import Generator
from utils.distributed import get_rank, synchronize, get_world_size

from src.train import train
from src.inference import inference
from src.evaluate import evaluate
from src.analysis import visualize_attention

def validate_args(args):
    '''
    Check some of the args and do some sanity checking
    We'll define default values here so that users don't need to 
    set them themselves. Reduce user burden, reduce user error.
    '''
    assert(args.type in ['train', 'inference', 'evaluate', 'attention_map'])
    arg_keys = args.keys()
    with open_dict(args):
        if "rank" not in arg_keys: args.rank = 0
        if "device" not in arg_keys: args.device = "cpu"
        if "world_size" not in arg_keys: args.world_size = 1
        if "rank" not in arg_keys: args.rank = 0
        if "local_rank" not in arg_keys: args.local_rank = 0
        if "distributed" not in arg_keys: #args.distributed = False
            if "WORLD_SIZE" in os.environ:
                # Single node multi GPU
                n_gpu = int(os.environ["WORLD_SIZE"])
            else:
                n_gpu = torch.cuda.device_count()
            args.distributed = n_gpu > 1
        if "workers" not in arg_keys: args.workers = 0
        # Validate training args
        if args.type == "train":
            # logging
            assert(hasattr(args, "logging"))
            if "print_freq" not in args.logging:
                args.logging.print_freq = 10000
            if "eval_freq" not in args.logging:
                args.logging.eval_freq = -1
            if "save_freq" not in args.logging:
                args.logging.save_freq = -1
            if "checkpoint_path" in args.logging:
                if args.logging.checkpoint_path[-1] != "/":
                    args.logging.checkpoint_path += "/"
            if "sample_path" in args.logging:
                if args.logging.sample_path[-1] != "/":
                    args.logging.sample_path += "/"
            if "reuse_samplepath" not in args.logging:
                args.logging.reuse_samplepath = False
        if args.type == "evaluation" or args.type == "train":
            assert(hasattr(args.evaluation, "gt_path")),f"You must specify "\
                    f"the ground truth data path"
            assert(hasattr(args.evaluation, "total_size")),f"You must specify "\
                    f"the number of images for FID"
            if "batch" not in args.evaluation:
                args.evaluation.batch = 1
            if "save_root" not in args:
                args.save_root = "/tmp/"
                if args.type == "training":
                    logging.warning("Save root path not set, using /tmp")
            if args.save_root[-1] != "/":
                args.save_root += "/"
        if args.type == "inference":
            if "batch" not in args.inference:
                args.inference.batch = 1
        #    assert(hasattr(args.evaluation, "gt_path"))
        if "misc" not in args.keys():
            args.misc = {}
            args.misc.seed = None
            args.misc.rng_state = None
            args.misc.py_rng_state = None
        else:
            if "seed" not in args.misc:
                args.misc.seed = None
            if "rng_state" not in args.misc:
                args.misc.rng_state= None
            if "py_rng_state" not in args.misc:
                args.misc.rng_state= None

def rng_reproducibility(args, ckpt=None):
    # Store RNG info
    # Cumbersome but reproducibility is not super easy
    with open_dict(args):
        if args.misc.seed is None:
            args.misc.seed = torch.initial_seed()
        else:
            torch.manual_seed(args.misc.seed)
        if args.misc.rng_state is None:
            args.misc.rng_state = torch.get_rng_state().tolist()
        else:
            torch.set_rng_state(args.misc.rng_state)
        if args.misc.py_rng_state is None:
            args.misc.py_rng_state = random.getstate()
        else:
            random.setstate(args.misc.py_rng_state)

    if ckpt is not None and "reuse_rng" in args.restart and args.restart.reuse_rng:
        with open_dict(args):
            if "misc" in ckpt['args'].keys() and "seed" in ckpt['args']['misc'].keys():
                args.misc.seed = ckpt['args']['misc']['seed']
            else:
                warnings.warn("No seed found in checkpoint arguments")
            if "misc" in ckpt['args'].keys() and "rng_state" in ckpt['args']['misc'].keys():
                args.misc.rng_state= ckpt['args']['misc']['rng_state']
            else:
                warnings.warn("No rng_state found in checkpoint arguments")
            if "misc" in ckpt['args'].keys() and "py_rng_state" in ckpt['args']['misc'].keys():
                args.misc.rng_state= ckpt['args']['misc']['py_rng_state']
            else:
                warnings.warn("No py_rng_state found in checkpoint arguments")
        try:
            torch.manual_seed(args.misc.seed)
            _seed = True
        except:
            warnings.warn("Unable to set manual_seed")
            _seed = False
        try:
            torch.set_rng_state(torch.as_tensor(
                ckpt['args']['misc']['rng_state'], dtype=torch.uint8),
            )
            _pt_rng = True
        except:
            warnings.warn("Unable to set ptyroch's rng state")
            _pt_rng = False
        try:
            l = tuple(ckpt['args']['misc']['py_rng_state'])
            random.setstate((l[0], tuple(l[1]), l[2]))
            _py_rng = True
        except:
            warnings.warn("Unable to set python's rng state")
            _py_rng = False
        print(f"RNG Loading Success States:\n"\
              f"\tSeed: {_seed}, PyTorch RNG: {_pt_rng}, Python RNG {_py_rng}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    if "logging_level" in args:
        if type(args.logging_level) == str:
            _logging_level = {"DEBUG": logging.DEBUG, "INFO":logging.INFO,
                    "WARNING": logging.WARNING, "ERROR": logging.ERROR,
                    "CRITICAL": logging.CRITICAL}[args.logging_level.upper()]
        else:
            _logging_level = int(args.logging_level)
        logging.getLogger().setLevel(_logging_level)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    validate_args(args)
    ckpt = None
    if "restart" in args and "ckpt" in args.restart and args.restart.ckpt:
        assert(os.path.exists(args.restart.ckpt)),f"Can't find a checkpoint "\
                f"at {args.restart.ckpt}"
        ckpt = torch.load(args.restart.ckpt, map_location=lambda storage, loc: storage)
        if "start_iter" not in args.restart:
            with open_dict(args):
                try:
                    args.restart.start_iter = \
                            int(os.path.basename(args.restart.ckpt)\
                            .split(".pt")[0])
                except:
                    args.restart.start_iter = 0

    rng_reproducibility(args, ckpt)
    #if "WORLD_SIZE" in os.environ:
    #    # Single node multi GPU
    #    n_gpu = int(os.environ["WORLD_SIZE"])
    #else:
    #    n_gpu = torch.cuda.device_count()
    #args.distributed = n_gpu > 1

    if args.distributed:
        try:
            args.local_rank = int(os.environ["LOCAL_RANK"])
        except:
            args.local_rank = 0
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://",
                                             timeout=timedelta(0, 180000))
        args.rank = get_rank()
        args.world_size = get_world_size()
        args.device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(args.local_rank)
        synchronize()

    if get_rank() == 0 and args.type in ['train']:
        # Make sample path
        if "sample_path" not in args.logging:
            samp_path = args.save_root + "samples"
        else:
            samp_path = args.logging.sample_path
            if args.logging.sample_path[0] != "/":
                samp_path = args.save_root + samp_path
        if not os.path.exists(samp_path):
            print(f"====> MAKING SAMPLE DIRECTORY: {samp_path}")
            os.mkdir(samp_path)
        # make checkpoint path
        if "checkpoint_path" not in args.logging:
            ckpt_path = args.save_root + "checkpoints"
        else:
            ckpt_path = args.logging.checkpoint_path
            if args.logging.checkpoint_path[0] != "/":
                ckpt_path = args.save_root + ckpt_path
        if not os.path.exists(ckpt_path):
            print(f"====> MAKING CHECKPOINT DIRECTORY: {cpkt_path}")
            os.mkdir(ckpt_path)


    # Only load gen if training, to save space
    if args.type == "train":
        generator = Generator(args=args.runs.generator, size=args.runs.size).to(args.device)
    g_ema = Generator(args=args.runs.generator, size=args.runs.size).to(args.device)

    if hasattr(g_ema, "num_params"):
        args.runs.generator.params = g_ema.num_params() / 1e6
    else:
        num_params = sum([m.numel() for m in g_ema.parameters()])
        if hasattr(args.runs.generator, "params"):
            args.runs.generator.params = num_params / 1e6
        else:
            with open_dict(args):
                args.runs.generator.params = num_params / 1e6

    # Load generator checkpoint
    if ckpt is not None:
        # Load generator checkpoints. But only load g if training
        if get_rank() == 0:
            print(f"Loading Generative Model")
        if 'state_dicts' in ckpt.keys():
            if args.type == "train":
                generator.load_state_dict(ckpt["state_dicts"]["g"])
            g_ema.load_state_dict(ckpt["state_dicts"]["g_ema"])
        elif set(['g', 'g_ema']).issubset(ckpt.keys()): # Old
            if args.type == "train":
                generator = Generator(args=args.runs.generator,
                        size=args.runs.size, legacy=True).to(args.device)
                generator.load_state_dict(ckpt['g'])
            g_ema = Generator(args=args.runs.generator,
                    size=args.runs.size, legacy=True).to(args.device)
            g_ema.load_state_dict(ckpt["g_ema"])
        else:
            raise ValueError(f"Checkpoint dict broken:\n"\
                    f"Checkpoint name: {args.restart.ckpt}\n"
                    f"Keys: {ckpt.keys()}")

    g_ema.eval()

    # Print mode in a nice format
    if get_rank() == 0:
        print("\n" + ("=" * 50))
        print(f" Mode: {args.type} ".center(49, "="))
        print("=" * 50, "\n")

    if args.type == "train":
        train(args=args,
              generator=generator,
              g_ema=g_ema,
              ckpt=ckpt,
              )
    elif args.type == "inference":
        inference(args=args, generator=g_ema)
    elif args.type == "evaluate":
        evaluate(args=args, generator=g_ema)
    elif args.type == "attention_map":
        visualize_attention(args, g_ema,
                            save_maps=args.evaluation.save_attn_map,
                            )


if __name__ == '__main__':
    main()
