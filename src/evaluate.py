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
    assert args.evaluation.save_path is not None,f"Inference save path needs to be defined"
    path = args.evaluation.save_path
    if steps is not None:
        path += f"eval_{str(steps)}"
        # We're only going to make the paths for training
        if not os.path.exists(path):
            os.mkdir(path)
    print(f"Saving {args.evaluation.num_batches * args.evaluation.batch} "\
          f"Images to {args.evaluation.save_path}")
    cnt = 0
    for _ in tqdm(range(args.evaluation.num_batches), desc="Saving Images"):
        noise = torch.randn((args.evaluation.batch, args.runs.generator.style_dim)).to(args.device)
        sample, _ = generator(noise)
        sample = unnormalize(sample)
        if args.logging.wandb and get_rank() == 0 and log_first_batch:
            grid = make_grid(sample, nrow=args.evaluation.batch)
            wandb.log({"samples": wandb.Image(grid)})
        Parallel(n_jobs=args.evaluation.batch)(delayed(save_image)
                (img, f"{path}/{str(cnt+j).zfill(6)}.png",
                nrow=1, padding=0, normalize=True, value_range=(0,1),)
                for j, img in enumerate(sample))
        cnt += args.evaluation.batch

@torch.no_grad()
def evaluate(args, generator, steps=None, log_first_batch=True):
    print(f" Parameters ".center(40, "-"))
    print(f"Generator has:".ljust(19),f"{args.runs.generator.params:.4f} M Parameters")
    assert args.evaluation.save_path is not None,f"Inference save path needs to be defined"
    path = args.evaluation.save_path
    if steps is not None:
        path += f"eval_{str(steps)}"
        # We're only going to make the paths for training
        if not os.path.exists(path):
            os.mkdir(path)
    # Save ALL the images
    save_images_batched(args=args,
                        generator=generator,
                        steps=steps,
                        log_first_batch=log_first_batch)
    if args.dataset.name in ['church']:
        dataset = get_dataset(args, evaluation=False)
        fid = fid_score.calculate_fid_path_and_dataset(path=path,
                                                       dataset=dataset,
                                                       batch_size=args.evaluation.batch,
                                                       device=args.device,
                                                       dims=2048,
                                                       num_workers=args.workers,
                                                       N=args.evaluation.total_size)
    else:
        gt_path = args.evaluation.gt_path
        fid = fid_score.calculate_fid_given_paths([path, gt_path],
                                                  batch_size=args.evaluation.batch,
                                                  device=args.device,
                                                  dims=2048,
                                                  num_workers=args.workers,
                                                  N=args.evaluation.total_size)
    print(f"{args.dataset.name} FID-{args.evaluation.total_size//1000}k : {fid:.3f}")
    return fid

