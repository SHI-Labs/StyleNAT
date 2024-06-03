from rich import print
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
from utils import helpers

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    #helpers.validate_args(args)
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

    #helpers.rng_reproducibility(args, ckpt)
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
