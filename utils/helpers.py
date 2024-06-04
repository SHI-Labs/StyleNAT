from typing import Any
import os
import random
import warnings
import logging
from omegaconf import OmegaConf, open_dict
import torch
from rich import print
# For legacy
import argparse 

def check_and_set_hydra(args, key : str, value : Any) -> None:
    if hasattr(args, key):
        args['key'] = value
    else:
        with open_dict(args):
            args['key'] = value
    logging.info(f"{args}.{key} = {value}")

def validate_args(args):
    '''
    Check some of the args and do some sanity checking
    We'll define default values here so that users don't need to 
    set them themselves. Reduce user burden, reduce user error.
    '''
    assert(args.type in ['train', 'inference', 'evaluate', 'attention_map'])
    arg_keys = args.keys()
    with open_dict(args):
        if "rank" not in arg_keys: check_and_set_hydra(args,"rank",0)
        if "device" not in arg_keys: check_and_set_hydra(args,"device","cpu")
        if "world_size" not in arg_keys: check_and_set_hydra(args,"world_size",1)
        if "rank" not in arg_keys: check_and_set_hydra(args,"rank",0)
        if "local_rank" not in arg_keys: check_and_set_hydra(args,"local_rank",0)
        if "distributed" not in arg_keys: #args.distributed = False
            if "WORLD_SIZE" in os.environ:
                # Single node multi GPU
                n_gpu = int(os.environ["WORLD_SIZE"])
            else:
                n_gpu = torch.cuda.device_count()
            check_and_set_hydra(args,"distributed",n_gpu > 1)
        if "workers" not in arg_keys: args.workers = 0
        # Validate training args
        if args.type == "train":
            # logging
            assert(hasattr(args, "logging"))
            if "print_freq" not in args.logging:
                check_and_set_hydra(args.logging,"print_freq",10000)
            if "eval_freq" not in args.logging:
                check_and_set_hydra(args.logging,"eval_freq",-1)
            if "save_freq" not in args.logging:
                check_and_set_hydra(args.logging,"save_freq",-1)
            if "checkpoint_path" in args.logging:
                if args.logging.checkpoint_path[-1] != "/":
                    args.logging.checkpoint_path += "/"
            else:
                check_and_set_hydra(args.logging,"checkpoint_path","./")
            if "sample_path" in args.logging:
                if args.logging.sample_path[-1] != "/":
                    args.logging.sample_path += "/"
            else:
                check_and_set_hydra(args.logging, "sample_path", "./")
            if "reuse_samplepath" not in args.logging:
                check_and_set_hydra(args.logging,"reuse_samplepath",False)
        if args.type == "evaluation" or args.type == "train":
            assert(hasattr(args.evaluation, "gt_path")),f"You must specify "\
                    f"the ground truth data path"
            assert(hasattr(args.evaluation, "total_size")),f"You must specify "\
                    f"the number of images for FID"
            if "batch" not in args.evaluation:
                check_and_set_hydra(args.evaluation,"batch", 1)
            if "save_root" not in args:
                check_and_set_hydra(args,"save_root","/tmp/")
                if args.type == "training":
                    logging.warning("Save root path not set, using /tmp")
            if args.save_root[-1] != "/":
                args.save_root += "/"
        if args.type == "inference":
            if "batch" not in args.inference:
                check_and_set_hydra(args.inference,"batch",1)
        #    assert(hasattr(args.evaluation, "gt_path"))
        if "misc" not in args.keys():
            check_and_set_hydra(args,"misc",{})
        if "seed" not in args.misc:
            check_and_set_hydra(args.misc,"seed",None)
        if "rng_state" not in args.misc:
            check_and_set_hydra(args.misc,"rng_state",None)
        if "py_rng_state" not in args.misc:
            check_and_set_hydra(args.misc,"rng_state",None)

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

    if ckpt is not None \
            and "reuse_rng" in args.restart \
            and args.restart.reuse_rng:
        with open_dict(args):
            #if "misc" in ckpt['args'].keys() and "seed" in ckpt['args']['misc'].keys():
            try:
                if hasattr(ckpt['args'], "misc"):
                    if hasattr(ckpt['args']['misc'], "seed"):
                        try:
                            args.misc.seed = ckpt['args']['misc']['seed']
                            print(f"[bold green]RNG Seed successfully loaded")
                        except:
                            print("[bold yellow]Seed couldn't be loaded (new style ckpt)")
                    else:
                        print("[bold yellow]Couldn't find seed (new style ckpt)")
                    if hasattr(ckpt['args']['misc'], 'rng_state'):
                        try:
                            args.misc.rng_state = ckpt['args']['misc']['rng_state']
                            print(f"[bold green]RNG State successfully loaded")
                        except:
                            print("[bold yellow]RNG State couldn't be loaded (new style ckpt)")
                    else:
                        print("[bold yellow]Couldn't find RNG State (new style ckpt)")
                    if hasattr(ckpt['args']['misc'], 'py_rng_state'):
                        try:
                            args.misc.py_rng_state = ckpt['args']['misc']['py_rng_state']
                            print(f"[bold green] Py-RNG State successfully loaded")
                        except:
                            print("[bold yellow]Py-RNG State couldn't be loaded (new style ckpt)")
                    else:
                        print("[bold yellow]Couldn't find Py-RNG State (new style ckpt)")
                elif type(ckpt['args']) == argparse.Namespace:
                    try:
                        args.misc.seed = ckpt['args'].seed
                        print(f"[bold green]RNG Seed successfully loaded")
                    except:
                        print("[bold yellow]Seed couldn't be loaded (old style ckpt)")
                    try:
                        args.misc.rng_state = ckpt['args'].rng_state.tolist()
                        print(f"[bold green]RNG State successfully loaded")
                    except:
                        print("[bold yellow]RNG State couldn't be loaded (old style ckpt)")
                    try:
                        args.misc.py_rng_state = ckpt['args'].rng_state.tolist()
                        print(f"[bold green] Py-RNG State successfully loaded")
                    except:
                        print("[bold yellow]Py-RNG State couldn't be loaded (old style ckpt)")
                else:
                    print("[bold yellow]No rng loading. {type(ckpt['args'])=}")
            except:
                print("[bold yellow]Seeds couldn't be loaded and don't know why")
                print(f"[bold yellow]\t {type(ckpt['args'])}")
                print(f"{type(ckpt['args']) == argparse.Namespace}")
        try:
            torch.manual_seed(args.misc.seed)
            _seed = "[bold green]True[/]"
        except:
            print("[bold yellow]Unable to set manual_seed")
            _seed = "[bold red]False[/]"
        try:
            torch.set_rng_state(torch.as_tensor(
                args.misc.rng_state, dtype=torch.uint8),
            )
            _pt_rng = "[bold green]True[/]"
        except:
            print("[bold yellow]Unable to set ptyroch's rng state")
            _pt_rng = "[bold red]False[/]"
        try:
            l = tuple(args.misc.py_rng_state)
            random.setstate((l[0], tuple(l[1]), l[2]))
            _py_rng = "[bold green]True[/]"
        except:
            print("[bold yellow]Unable to set python's rng state")
            _py_rng = "[bold red]False[/]"
        print(f"[bold green]RNG Loading Success States:[/]\n"\
              f"\tSeed: {_seed}, PyTorch RNG: {_pt_rng}, Python RNG {_py_rng}")
