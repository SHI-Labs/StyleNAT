import time
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import wandb
import torch
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

from dataset.dataset import unnormalize
from utils import fid_score
from utils.improved_precision_recall import IPR
from utils.distributed import get_rank

@torch.no_grad()
def save_images_batched(args, generator, steps=None, log_first_batch=True):
    if args.logging.sample_path[0] != "/":
        path = args.save_path + args.logging.sample_path
    else:
        path = args.logging.sample_path
    if steps is not None:
        path += f"eval_{str(steps)}"
        # We're only going to make the paths for training
        if not os.path.exists(path):
            os.mkdir(path)
    assert(args.evaluation.total_size % args.evaluation.batch == 0),\
            f"Evaluation total size % batch should be zero. Got "\
            f"{args.evaluation.total_size % args.evaluation.batch}"
    print(f"Saving {args.evaluation.total_size} "\
          f"Images to {path}")
    cnt = 0
    nbatches = args.evaluation.total_size // args.evaluation.batch
    for _ in tqdm(range(nbatches), desc="Saving Images"):
        noise = torch.randn((args.evaluation.batch, args.runs.generator.style_dim)).to(args.device)
        sample, _ = generator(noise)
        sample = unnormalize(sample)
        if args.logging.wandb and get_rank() == 0 and log_first_batch and cnt == 0:
            grid = make_grid(sample, nrow=args.evaluation.batch)
            wandb.log({"samples": wandb.Image(grid)}, commit=False)
        Parallel(n_jobs=args.evaluation.batch)(delayed(save_image)
                (img, f"{path}/{str(cnt+j).zfill(6)}.png",
                nrow=1, padding=0, normalize=True, value_range=(0,1),)
                for j, img in enumerate(sample))
        cnt += args.evaluation.batch

@torch.no_grad()
def clear_directory(args):
    if args.logging.sample_path[0] != "/":
        path = args.save_path + args.logging.sample_path
    else:
        path = args.logging.sample_path
    _files = os.listdir(path)
    if _files == []: 
        print(f"Directory {path} is already empty. Worry if not first time")
        return
    assert(".png" in _files[0])
    Parallel(n_jobs=32)(delayed(os.remove)
            (path + img) for img in _files)


@torch.no_grad()
def evaluate(args,
             generator,            # Should be g_ema
             steps=None,           # Save to specific eval dir or common
             log_first_batch=True, # Log first batch of images to wandb?
             ):
    #print(f" Parameters ".center(40, "-"))
    print(f"> Generator has:".ljust(19),f"{args.runs.generator.params:.4f} M Parameters")
    if not hasattr(args.logging, "sample_path"):
        path = args.save_path
    else:
        if args.logging.sample_path[0] != "/":
            path = args.save_path+ args.logging.sample_path
        else:
            path = args.logging.sample_path
    assert(type(path) == str),f"Path needs to be a string not {type(path)}"
    if path[-1] != "/": path = path + "/"
    if steps is not None:
        path += f"eval_{str(steps)}"
        # We're only going to make the paths for training
        if not os.path.exists(path):
            os.mkdir(path)
    # Save ALL the images
    if args.logging.reuse_samplepath:
        clear_directory(args)
    torch.cuda.synchronize()
    save_images_batched(args=args,
                        generator=generator,
                        steps=steps,
                        log_first_batch=log_first_batch)
    if args.evaluation.gt_path[0] != "/":
        gt_path = args.dataset.path + args.evaluation.gt_path
    else:
        gt_path = args.evaluation.gt_path
    fid = fid_score.calculate_fid_given_paths([path, gt_path],
                                              batch_size=args.evaluation.batch,
                                              device=args.device,
                                              dims=2048,
                                              num_workers=args.workers,
                                              N=args.evaluation.total_size)
    print(f"{args.dataset.name} FID-{args.evaluation.total_size//1000}k : "\
          f"{fid:.3f} for {steps} steps")
    return fid

