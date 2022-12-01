import time
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F

from utils.distributed import get_rank, synchronize, get_world_size, reduce_loss_dict
from utils.CRDiffAug import CR_DiffAug
from dataset.dataset import get_dataset, get_loader
from models.discriminator import Discriminator

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
    dataset = get_dataset(args, evaluation=False)
    if get_rank() == 0:
        print(f"="*50)
        print(f" Dataset: {args.dataset.name} ".center(50, "="))
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
    if args.restart.ckpt:
        # Check starting iteration
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            if get_rank() == 0:
                print(f"Starting from {args.start_iter} iters")
        except:
            if get_rank() == 0:
                print(f"Starting from 0 iters")

        # Load discriminator state dict
        try:
            discriminator.load_state_dict(ckpt["d"])
        except:
            if get_rank() == 0:
                print("We don't load the discriminator!")

    # Define all args above this point so we properly save to wandb
    if get_rank() == 0 and wandb is not None and args.logging.wandb:
        wandb.init(project=args.wandb.project_name,
                   entity=args.wandb.entity,
                   name=args.wandb.run_name,
                   config=args,
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
    if args.restart.ckpt is not None:
        if get_rank() == 0:
            print("loading optimizer: ", args.restart.ckpt)

        try:
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
    for idx in range(args.runs.training.iter):
        i = idx + args.restart.start_iter
        if i > args.runs.training.iter:
            if get_rank() == 0:
                print("Done!")
            break

        # Train D
        generator.train()
        if not args.dataset.lmdb:
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

        if not args.dataset.lmdb:
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
            args.G_lr -= lr_decay_per_step
            args.D_lr = args.G_lr * 4 if args.ttur else (args.D_lr - lr_decay_per_step)

            for param_group in d_optim.param_groups:
                param_group['lr'] = args.runs.training.discriminator.lr
            for param_group in g_optim.param_groups:
                param_group['lr'] = args.runs.training.generator.lr

        # Log, save, and evaluate
        if get_rank() == 0:
            if i % args.logging.print_freq == 0:
                vis_loss = {
                    'd_loss': d_loss_val,
                    'g_loss': g_loss_val,
                    'r1_val': r1_val,
                    }
                if wandb and args.logging.wandb:
                    wandb.log(vis_loss, step=i)
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

            if i != 0 and i % args.logging.eval_freq == 0:
                print(f"Saving model at: "\
                      f"{args.logging.checkpoint_path}/{str(i).zfill(6)}.pt")
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    args.logging.checkpoint_path + f"/{str(i).zfill(6)}.pt",
                )

                # Evaluate with EMA and log the FID score
                print("===> Evaluation <===")
                g_ema.eval()
                fid = evaluation(g_ema, args, i * args.runs.training.batch * int(os.environ["WORLD_SIZE"]))
                steps = i * args.runs.training.batch * int(os.environ["WORLD_SIZE"])
                print(f"FID Score : {fid:.2f}, {steps/1000000:.1f}M images processed")
                fid_dict = {'fid': fid}
                if wandb and args.logging.wandb:
                    wandb.log({'fid': fid,
                               'iter': i})

            if i % args.logging.save_freq == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "G_lr": args.runs.generator.lr,
                        "D_lr": args.runs.discriminator.lr,
                    },
                    args.logging.checkpoint_path + f"/{str(i).zfill(6)}.pt",
                )
        torch.cuda.synchronize()
