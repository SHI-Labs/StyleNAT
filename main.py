import hydra
import os
from datetime import timedelta
import torch

from models.generator import Generator
from utils.distributed import get_rank, synchronize, get_world_size

from src.train import train
from src.inference import inference
from src.evaluate import evaluate
from src.interpolation import interpolate
from src.inversion import inversion
from src.stylemc import stylemc


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):


    # Store RNG info
    if args.misc.seed is None:
        args.misc.seed = torch.initial_seed()
        args.misc.rng_state = torch.get_rng_state().tolist()
        # For verbose
        #print(f"Created new seed!")
        #print(f"Seed: {args.misc.seed} and state: {torch.get_rng_state()}")
    else:
        torch.manual_seed(args.misc.seed)
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(f"Using MANUAL SEED: {args.misc.seed}")

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://",
                                             timeout=timedelta(0, 180000))
        args.rank = get_rank()
        args.world_size = get_world_size()
        args.device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(args.local_rank)
        synchronize()

    if get_rank() == 0:
        args.logging.sample_path = os.path.join(args.logging.sample_path, 'samples')
        if not os.path.exists(args.logging.sample_path):
            os.mkdir(args.logging.sample_path)


    generator = Generator(args=args.runs.generator, size=args.runs.size).to(args.device)
    g_ema = Generator(args=args.runs.generator, size=args.runs.size).to(args.device)

    if hasattr(generator, "num_params"):
        args.runs.generator.params = generator.num_params() / 1e6
    else:
        num_params = sum([m.numel() for m in generator.parameters()])
        args.runs.generator.params = num_params / 1e6

    # Load generator checkpoint
    ckpt = None
    if args.restart.ckpt:
        if get_rank() == 0:
            print(f"Loading model: {args.restart.ckpt}")
        ckpt = torch.load(args.restart.ckpt, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(args.restart.ckpt)
        generator.load_state_dict(ckpt['g'])
        g_ema.load_state_dict(ckpt["g_ema"])


    g_ema.eval()

    # Print mode in a nice format
    if get_rank() == 0:
        print("\n" + ("=" * 50))
        print(f" Mode: {args.type} ".center(50, "="))
        print("=" * 50, "\n")

    if args.type == "train":
        train(args=args,
              generator=generator,
              g_ema=g_ema,
              ckpt=ckpt,
              )
    elif args.type == "inference":
        inference(args=args, generator=g_ema)
    elif args.type == "interpolate":
        interpolate(args=args, generator=g_ema)
    elif args.type == "evaluate":
        evaluate(args=args, generator=g_ema)
    elif args.type == "inversion":
        inversion(args=args, generator=g_ema)
    elif args.type == "stylemc":
        stylemc(args=args, generator=g_ema)


if __name__ == '__main__':
    main()
