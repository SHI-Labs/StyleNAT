from rich import print
import time
import os
import string
import random
import warnings
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from utils.distributed import get_rank, synchronize, get_world_size, reduce_loss_dict
from utils.CRDiffAug import CR_DiffAug
from dataset.dataset import get_dataset, get_loader
from models.discriminator import Discriminator
from src.evaluate import evaluate
from src.analysis import visualize_attention

try:
    import wandb
except ImportError:
    wandb = None

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def d_logistic_loss(real_pred, fake_pred):
    assert type(real_pred) == type(fake_pred), "real_pred must be the same type as fake_pred"
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss



def train(args, generator, g_ema, ckpt=None):
    # Preprocessing checks and sets
    if wandb:
        wandb_dict = {}
    save_dict, eval_dict = {}, {}
    dataset = get_dataset(args, evaluation=False)
    if not hasattr(args, "rank"):
        with open_dict(args):
            args.rank = 0
    if not hasattr(args, "world_size"):
        with open_dict(args):
            args.world_size = 1
    if get_rank() == 0:
        print(f"="*50)
        print(f" Dataset: {args.dataset.name} ".center(49, "="))
        print(f"="*50)
    loader = get_loader(args=args,
                        dataset=dataset,
                        batch_size=args.runs.training.batch)


    # Load discriminator
    discriminator = Discriminator(args=args.runs.discriminator,
                                  size=args.runs.size).to(args.device)

    if hasattr(discriminator, "num_params"):
        args.runs.discriminator.params = discriminator.num_params() / 1e6
    else:
        num_params = sum([m.numel() for m in discriminator.parameters()])
        if hasattr(args.runs.discriminator, "params"):
            args.runs.discriminator.params = num_params / 1e6
        else:
            with open_dict(args):
                args.runs.discriminator.params = num_params / 1e6

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
    if ckpt is not None:
        # Load discriminator state dict
        try:
            if "state_dicts" in ckpt.keys():
                discriminator.load_state_dict(ckpt["state_dicts"]["d"])
            else: # Old
                discriminator.load_state_dict(ckpt["d"])
        except:
            if get_rank() == 0:
                print("We don't load the discriminator!")

    # Define all args above this point so we properly save to wandb
    # Don't worry, other args will be logged with the checkpoints
    if get_rank() == 0 and wandb is not None and args.logging.wandb:
        assert(hasattr(args, "wandb")),f"arg has ({args.keys()}) but not wandb?"
        if hasattr(args.wandb, "tags"):
            tags = args.wandb.tags
            print(f"Using tags {tags}")
        else:
            tags = None
        if hasattr(args.wandb, "description"):
            description = args.wandb.description
            print(f"Description".center(40,"="))
            print("\t",description.replace('\n','\n\t'))
            print(f"".center(40,"="))
        else:
            description = None
        _resume = None
        _id = None
        if "restart" in args:
            if"wandb_resume" in args.restart:
                _resume = args.restart.wandb_resume
            if "wandb_id" in args.restart:
                _id = args.restart.wandb_id
        wandb.init(project=args.wandb.project_name,
                   entity=args.wandb.entity,
                   name=args.wandb.run_name,
                   config=OmegaConf.to_container(args,
                                                 resolve=True,
                                                 throw_on_missing=True),
                   resume=_resume,
                   id=_id,
                   notes=description,
                   tags=tags,
                )

    ## VERBOSE
    #print("-" * 80)
    #print("Generator: ")
    ##print(generator)
    #print("-" * 80)
    if args.runs.generator.block_type == 'nat' and get_rank() == 0:
        print(f"Using NA blocks with"\
              f"\nKernels: \n {generator.kernels}"\
              f"\nDilations: \n {generator.dilations}")

    # Distributed
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True, 
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # Optimizers
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
    if ckpt:
        if get_rank() == 0:
            print("loading optimizer: ")#, args.restart.ckpt)

        try:
            if "optim_dicts" in ckpt.keys():
                g_optim.load_state_dict(ckpt["optim_dicts"]["g_optim"])
            else: # Old
                g_optim.load_state_dict(ckpt["g_optim"])
        except:
            if get_rank() == 0:
                print(f"We didn't load the generator's optimizer!")
        try:
            d_optim.load_state_dict(ckpt["d_optim"])
        except:
            if get_rank() == 0:
                print("We don't load the discriminator's optimizer.")


    # Print parameters
    if get_rank() == 0:
        print(f" Parameters ".center(40, "-"))
        print(f"Generator has:".ljust(19),f"{args.runs.generator.params:.4f} M Parameters")
        print(f"Discriminator has:".ljust(19),f"{args.runs.discriminator.params:.4f} M Parameters")

    # Log the base learning rates 
    G_lr_base = args.runs.generator.lr
    D_lr_base = args.runs.discriminator.lr

    loader = sample_data(loader)

    # Loss Setup
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=args.device)
    g_loss_val = 0
    accum = 0.5 ** (32 / (10 * 1000))
    loss_dict = {}
    l2_loss = torch.nn.MSELoss()
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    if get_rank() == 0:
        print(f" Start Training ".center(20, '-'))
    end = time.time()

    # ttur and lr decay
    if args.runs.training.ttur:
        args.runs.generator.lr = args.runs.discriminator.lr / 4
    if args.runs.training.lr_decay:
        lr_decay_per_step = args.runs.generator.lr / (args.runs.training.iter \
                - args.runs.training.lr_decay_start_steps)

    # Training loop
    start = 0
    if ckpt and "start_iter" in args.restart:
        start += args.restart.start_iter
        if get_rank() == 0:
            print(f"=====> Found checkpoint and starting at {start} iters")
    for i in range(start,args.runs.training.iter):
        if i > args.runs.training.iter:
            if get_rank() == 0:
                print("Done!")
            break

        # Train D
        generator.train()
        if "lmdb" not in args.dataset or not args.dataset.lmdb:
            this_data = next(loader)
            real_img = this_data[0]
        else:
            real_img = next(loader)
        real_img = real_img.to(args.device)

        generator.requires_grad = False
        discriminator.requires_grad = True
        noise = torch.randn((args.runs.training.batch, 512)).cuda()

        fake_img, _ = generator(noise)

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred) \
                 * args.runs.training.gan_weight

        if args.runs.training.bcr:
            real_img_cr_aug = CR_DiffAug(real_img)
            fake_img_cr_aug = CR_DiffAug(fake_img)
            fake_pred_aug = discriminator(fake_img_cr_aug)
            real_pred_aug = discriminator(real_img_cr_aug)
            d_loss += args.runs.training.bcr_fake_lambda \
                    * l2_loss(fake_pred_aug, fake_pred) \
                    + args.runs.training.bcr_real_lambda \
                    * l2_loss(real_pred_aug, real_pred)

        loss_dict["d"] = d_loss

        discriminator.zero_grad(set_to_none=True)
        d_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        d_optim.step()

        d_regularize = i % args.runs.discriminator.reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad(set_to_none=True)
            (args.runs.training.gan_weight \
                    * (args.runs.training.r1 / 2 \
                    * r1_loss \
                    * args.runs.discriminator.reg_every \
                    + 0 * real_pred[0])).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # Train G
        generator.requires_grad = True
        discriminator.requires_grad = False

        if "lmdb" not in args.dataset or not args.dataset.lmdb:
            this_data = next(loader)
            real_img = this_data[0]
        else:
            real_img = next(loader)
            real_img = real_img.to(device)

        noise = torch.randn((args.runs.training.batch,
                             args.runs.generator.style_dim)).cuda()
        fake_img, _ = generator(noise)
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)* args.runs.training.gan_weight

        loss_dict["g"] = g_loss
        generator.zero_grad(set_to_none=True)
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        # Finish one iteration and reduce loss dict
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()


        if args.runs.training.lr_decay \
                and i > args.runs.training.lr_decay_start_steps:
            args.runs.generator.lr -= lr_decay_per_step
            args.runs.discriminator.lr = args.runs.generator.lr * 4 if args.runs.training.ttur else (args.runs.discriminator.lr - lr_decay_per_step)

            for param_group in d_optim.param_groups:
                param_group['lr'] = args.runs.discriminator.lr
            for param_group in g_optim.param_groups:
                param_group['lr'] = args.runs.generator.lr

        # Log, save, and evaluate
        if get_rank() == 0:
            args.world_size = get_world_size()
            args.rank = get_rank()
            imgs_processed = i * args.runs.training.batch * get_world_size()
            if args.logging.checkpoint_path[0] != "/":
                ckpt_path = args.save_root + args.logging.checkpoint_path
            else:
                ckpt_path = args.logging.checkpoint_path
            save_name = f"{ckpt_path}/{str(i).zfill(6)}"
            # Don't overwrite existing checkpoint!
            if os.path.exists(f"{save_name}.pt"):
                _hash = "".join(random.choices(
                    string.ascii_uppercase + string.ascii_lowercase \
                    + string.digits, k=4))
                _save_name = f"{save_name}_{_hash}"
                while os.path.exists(f"{_save_name}.pt"):
                    _hash = "".join(random.choices(
                        string.ascii_uppercase + string.ascii_lowercase \
                        + string.digits, k=4))
                    _save_name = f"{save_name}_{_hash}"
                save_name = _save_name
                warnings.warn(f"\n==========> WARNING <==========\n"\
                        f"\tWe found an existing checkpoint so the "\
                        f"current run is saved as {save_name}" \
                        f".pt to avoid loss of data.\n"\
                        f"Please check, you may have multiple backups!\n"
                        f"===============================\n")
            if i % args.logging.print_freq == 0:
                if wandb and args.logging.wandb:
                    wandb_dict.update({
                    'd_loss': d_loss_val,
                    'g_loss': g_loss_val,
                    'r1_val': r1_val,
                    'iter': i,
                    'imgs_processed': imgs_processed,
                    "G_lr": args.runs.generator.lr,
                    "D_lr": args.runs.discriminator.lr,
                    })
                iters_time = time.time() - end
                end = time.time()
                if args.runs.training.lr_decay:
                    print(f"Iters: {i}"\
                          f"\tTime: {iters_time:.4f}"\
                          f"\tD_loss: {d_loss_val:.4f}"\
                          f"\tG_loss: {g_loss_val:.4f}"\
                          f"\tR1: {r1_val:.4f}"\
                          f"\tG_lr: {args.runs.generator.lr}"\
                          f"\tD_lr: {args.runs.discriminator.lr}")
                else:
                    print(f"Iters: {i}"\
                          f"\tTime: {iters_time:.4f}"\
                          f"\tD_loss: {d_loss_val:.4f}"\
                          f"\tG_loss: {g_loss_val:.4f}"\
                          f"\tR1: {r1_val:.4f}")
            if i % args.logging.save_freq == 0 and i != 0:
                state_dicts = {"g": g_module.state_dict(),
                               "d": d_module.state_dict(),
                               "g_ema": g_ema.state_dict(),
                              }
                optim_dicts = {"g_optim": g_optim.state_dict(),
                               "d_optim": d_optim.state_dict(),
                              }
                # LR should populate via optim, this is purposeful redundancy
                lr_dicts = {"G_lr": args.runs.generator.lr,
                            "D_lr": args.runs.discriminator.lr,
                           }
                # Convert args back to normal dict to make exploring checkpoint
                # easier
                save_dict.update({
                        "args": OmegaConf.to_container(args, resolve=True),
                        "state_dicts": state_dicts,
                        "optim_dicts": optim_dicts,
                        "lr_dicts": lr_dicts,
                    })

            if i != 0 and i % args.logging.eval_freq == 0:
                # Evaluate with EMA and log the FID score
                print("===> Evaluation <===")
                g_ema.eval()
                # quick save since fid and analysis is most likely point for 
                # crash  We'll overwrite this checkpoint but want to provide 
                # robustness here to avoid unnecessary compute
                torch.save(save_dict, f"{save_name}.pt")
                print(f"Saved checkpoint to {save_name}.pt")
                fid = evaluate(args,
                               generator=g_ema,
                               steps=None if args.logging.reuse_samplepath else imgs_processed,
                               log_first_batch=args.logging.log_img_batch,
                               )
                # Save evaluations between each saved checkpoint, but record
                # when that was
                eval_dict.update({f"fid @ {imgs_processed/1e6:.2f}M imgs": fid,
                                  f"{imgs_processed/1e6:.2f}M imgs is": \
                                        f"{i} * {args.runs.training.batch} "\
                                        f"* {get_world_size()}",
                                 })
                print("="*50)
                print(f"> FID Score : {fid:.2f}, {imgs_processed/1000000:.2f}M "\
                      f"imgs seen <".center(48, '='))
                print("="*50)
                if wandb and args.logging.wandb:
                    wandb_dict.update({'fid': fid})
                    # If we aren't in our print frequency, just push to wandb 
                    # now. Only happens if eval % print != 0
                    if args.logging.print_freq != 0:
                        wandb.log(wandb_dict, step=i)
                        wandb_dict = {}

                # Log information to our checkpoint
                if hasattr(args.logging, "fid"):
                    args.logging.fid = float(fid)
                else:
                    with open_dict(args):
                        args.logging.fid = float(fid)
                if hasattr(args.logging, "current_iteration"):
                    args.logging.current_iteration= int(i)
                else:
                    with open_dict(args):
                        args.logging.current_iteration= int(i)
                if hasattr(args.logging, "nimages"):
                    args.logging.nimages= int(imgs_processed)
                else:
                    with open_dict(args):
                        args.logging.nimages=int(imgs_processed)

                if args.evaluation.attn_map:
                    visualize_attention(args, g_ema,
                                        save_maps=args.evaluation.save_attn_map,
                                        log_wandb=args.logging.wandb,
                                        )

            if i % args.logging.save_freq == 0 and i != 0:
                # Convert args back to normal dict to make exploring checkpoint
                # easier
                save_dict.update({
                        "args": OmegaConf.to_container(args, resolve=True),
                        "eval_info": eval_dict,
                    })
                torch.save(save_dict, f"{save_name}.pt")
                print(f"Saved checkpoint to {save_name}.pt")
                # Clear save_dict and eval_dict after save
                save_dict, eval_dict = {}, {}
            if wandb_dict != {}:
                wandb.log(wandb_dict,
                          commit=True,
                          step=i)
                # Clear wandb_dict after save
                wandb_dict = {}
        torch.cuda.synchronize()
