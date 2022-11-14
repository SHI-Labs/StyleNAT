import hydra
import builtins
import time
import os
import sys
from datetime import timedelta
from tqdm import tqdm
import math
import numpy as np
from joblib import Parallel, delayed

import torch
import torchvision
import torchvision.datasets as datasets
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid, save_image

try:
    import wandb
except ImportError:
    wandb = None

import time

from dataset.dataset import MultiResolutionDataset
from models.discriminator import Discriminator
from models.generator import Generator
from utils.CRDiffAug import CR_DiffAug
from utils.distributed import get_rank, reduce_loss_dict, synchronize, get_world_size

from src.inference import inference
from src.evaluate import evaluate

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):


    # Store RNG info
    if args.misc.seed is None:
        args.misc.seed = torch.initial_seed()
        args.misc.rng_state = torch.get_rng_state().tolist()
        print(f"Created new seed!")
        print(f"Seed: {args.misc.seed} and state: {torch.get_rng_state()}")
    else:
        torch.manual_seed(args.misc.seed)
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

    if args.distributed and get_rank() != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if get_rank() == 0:
        args.logging.sample_path = os.path.join(args.logging.sample_path, 'samples')
        if not os.path.exists(args.logging.sample_path):
            os.mkdir(args.logging.sample_path)


    generator = Generator(args=args.runs.generator, size=args.runs.size).to(args.device)

    if hasattr(generator, "num_params"):
        args.runs.generator.params = generator.num_params() / 1e6
    else:
        num_params = sum([m.numel() for m in generator.parameters()])
        args.runs.generator.params = num_params / 1e6

    discriminator = Discriminator(args=args.runs.discriminator, size=args.runs.size).to(args.device)

    if hasattr(discriminator, "num_params"):
        args.runs.discriminator.params = discriminator.num_params() / 1e6
    else:
        num_params = sum([m.numel() for m in discriminator.parameters()])
        args.runs.discriminator.params = num_params / 1e6

    g_ema = Generator(args=args.runs.generator, size=args.runs.size).to(args.device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    if args.runs.generator.reg_every is not None:
        g_reg_ratio = float(args.runs.generator.reg_every) / float(args.runs.generator.reg_every + 1)
    else:
        g_reg_ratio = 1
    if args.runs.discriminator.reg_every is not None:
        d_reg_ratio = float(args.runs.discriminator.reg_every) / float(args.runs.discriminator.reg_every + 1)
    else:
        g_reg_ratio = 1

    # Load model checkpoint.
    if args.restart.ckpt:
        print("load model: ", args.restart.ckpt)
        ckpt = torch.load(args.restart.ckpt, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(args.restart.ckpt)
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            pass

        generator.load_state_dict(ckpt["g"])
        g_ema.load_state_dict(ckpt["g_ema"])
        try:
            discriminator.load_state_dict(ckpt["d"])
        except:
            print("We don't load D.")

    # Define all args above this point so we properly save to wandb
    if get_rank() == 0 and wandb is not None and args.logging.wandb:
        wandb.init(project=args.wandb.project_name,
                   entity=args.wandb.entity,
                   name=args.wandb.run_name,
                   config=args,
                )

    print("-" * 80)
    print("Generator: ")
    #print(generator)
    print("-" * 80)
    if args.runs.generator.block_type == 'nat':
        print(f"Using NA blocks with"\
              f"\nKernels: \n {generator.kernels}"\
              f"\nDilations: \n {generator.dilations}")

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.runs.generator.lr * g_reg_ratio if not args.runs.training.ttur else args.runs.discriminator.lr / 4 * g_reg_ratio,
        betas=(args.runs.training.beta1 ** g_reg_ratio, args.runs.training.beta2 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.runs.discriminator.lr * d_reg_ratio,
        betas=(args.runs.training.beta1 ** d_reg_ratio, args.runs.training.beta2 ** d_reg_ratio),
    )

    # Load optimizer checkpoint.
    if args.restart.ckpt is not None:
        print("load optimizer: ", args.restart.ckpt)
        ckpt = torch.load(args.restart.ckpt, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(args.restart.ckpt)

        try:
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])
        except:
            print("We don't load optimizers.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = [transforms.Resize((args.runs.size, args.runs.size))]
    if args.runs.training.use_flip:
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    transform = transforms.Compose(transform)

    if args.dataset.lmdb:
        print(f"Using lmdb with {args.dataset.path}")
        dataset = MultiResolutionDataset(args.dataset.path, transform, args.runs.size)
    elif args.dataset.name in ['church']:
        from torchvision.datasets import LSUN
        if args.dataset.name == 'church':
            classes = ['church_outdoor_train']
        print(f"Loading LSUN {classes}")
        dataset = LSUN(root=args.dataset.path, transform=transform, classes=classes)
    elif args.dataset.name == 'cifar10':
        print(f"Loading CIFAR10")
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root=args.dataset.path, transform=transform)
    else:
        print(f"Loading ImageFolder dataset from {args.dataset.path}")
        dataset = datasets.ImageFolder(root=args.dataset.path, transform=transform)
    #dataset = datasets.CIFAR10(root="/data/datasets/", download=True, transform=transform)

    loader = data.DataLoader(
        dataset,
        batch_size=args.runs.training.batch,
        num_workers=args.runs.training.workers,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    print(f"Using {args.dataset.name} dataset")
    print(f"Generator has {args.runs.generator.params} M Parameters and "\
          f"Discriminator has {args.runs.discriminator.params} M Parameters")
    print(f"Args type {args.type}")

    if args.type == "inference":
        inference(args, g_ema, dataset)
    elif args.type == "evaluate":
        evaluate(args=args, generator=generator)



if __name__ == '__main__':
    main()
