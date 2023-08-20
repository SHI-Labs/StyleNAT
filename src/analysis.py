import os 
import numpy as np
import math
import random
import logging
from omegaconf import open_dict
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from matplotlib import cm
import wandb

from dataset.dataset import unnormalize
from natten.functional import natten2dqkrpb, natten2dav

def analysis(args, generator):
    if args.analysis.type.lower() == "attention":
        visualize_attention(args, generator)


def attn_wrapper(attn_object,
                 block_name, # Name of block
                 ):
    allowed_block_types = ("mhsarpb", "neighborhoodattentionsplithead", "hydraneighborhoodattention", "windowattention")
    block_name = block_name.lower()
    assert(block_name in allowed_block_types),f"Block Name {block_name} not supported: allowed = {allowed_block_types}"
    def na_fwd_hook(x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        qkv = attn_object.qkv(x).reshape(B, H, W, 3, attn_object.num_heads, attn_object.head_dim).permute(3, 0, 4, 1, 2, 5)

        q, k, v = qkv.chunk(3, dim=0)
        q = q.squeeze(0) * attn_object.scale
        k = k.squeeze(0)
        v = v.squeeze(0)

        if attn_object.clean_partition:
            q = q.chunk(attn_object.num_splits, dim=1)
            k = k.chunk(attn_object.num_splits, dim=1)
            v = v.chunk(attn_object.num_splits, dim=1)
        else:
            i = 0
            _q = []
            _k = []
            _v = []
            for h in attn_object.shapes:
                _q.append(q[:, i:i+h, :, :])
                _k.append(k[:, i:i+h, :, :])
                _v.append(v[:, i:i+h, :, :])
                i = i+h
            q, k, v = _q, _k, _v


        attention = [natten2dqkrpb(_q, _k, _rpb, _kernel_size, _dilation) for \
                _q,_k,_rpb,_kernel_size, _dilation in \
                zip(q, k, attn_object.rpb,
                    attn_object.kernel_sizes, attn_object.dilations)]
        attention = [a.softmax(dim=-1) for a in attention]
        ############################
        #attn_object.attn_map = attention
        logging.debug(f"NA fwd hook: q is type {type(q)}, and length {len(q)}")
        logging.debug(f"NA fwd hook: q shape is {[_q.shape for _q in q]}")
        qq = torch.cat(q, dim=1)
        kk = torch.cat(k, dim=1)
        qq = qq.mean([2,3]).unsqueeze(2)
        kk = kk.flatten(2,3).transpose(-2, -1)

        aa = qq @ kk
        aa = aa.reshape(B, attn_object.num_heads, H, W)
        attn_object.attn_map = aa
        ############################
        attention = [attn_object.attn_drop(a) for a in attention]

        x = [natten2dav(_attn, _v, _k, _d) for _attn, _v, _k, _d in \
                zip(attention, v, attn_object.kernel_sizes, attn_object.dilations)]

        x = torch.cat(x, dim=1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        return attn_object.proj_drop(attn_object.proj(x))

    def na_legacy_fwd_hook(x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        qkv = attn_object.qkv(x).reshape(B, H, W, 3, attn_object.num_heads, attn_object.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * attn_object.scale

        # Split along heads
        q0, q1 = q.chunk(chunks=2, dim=1)
        k0, k1 = k.chunk(chunks=2, dim=1)
        v0, v1 = v.chunk(chunks=2, dim=1)

        # TODO: fix natten signature for legacy
        attn0 = natten2dqkrpb(q0, k0, attn_object.rpb0, attn_object.dilation_0)
        attn0 = attn0.softmax(dim=-1)
        attn0_ = attn_object.attn_drop(attn0)

        x0 = natten2dav(attn0_, v0, attn_object.dilation_0)

        # TODO: fix natten signature for legacy
        attn1 = natten2dqkrpb(q1, k1, attn_object.rpb1, attn_object.dilation_1)
        attn1 = attn1.softmax(dim=-1)
        attn1_ = attn_object.attn_drop(attn1)

        ############################
        #attn_object.attn_map = torch.cat([attn0, attn1], dim=1)
        qq = q.mean([2, 3]).unsqueeze(2)
        kk = k.flatten(2, 3).transpose(-2, -1)
        aa = qq @ kk
        aa = aa.reshape(B, attn_object.num_heads, H, W)
        attn_object.attn_map = aa
        logging.debug(f"NA legacy fwd hook: q,k,a shapes "\
                f"{qq.shape}, {kk.shape}, {aa.shape}")
        ############################

        x1 = natten2dav(attn1_, v1, attn_object.dilation_1)

        x = torch.cat([x0, x1],dim=1)

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        return attn_object.proj_drop(attn_object.proj(x))
    def swin_fwd_hook(q, k, v, mask=None):
        B_, N, C = q.shape
        q = q.reshape(B_, N, attn_object.num_heads, C // attn_object.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, attn_object.num_heads, C // attn_object.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, attn_object.num_heads, C // attn_object.num_heads).permute(0, 2, 1, 3)
        # B_, num_heads, N, head_dim

        q = q * attn_object.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = attn_object.relative_position_bias_table[attn_object.relative_position_index.view(-1)].view(
            attn_object.window_size[0] * attn_object.window_size[1], attn_object.window_size[0] * attn_object.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, attn_object.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, attn_object.num_heads, N, N)
            attn = attn_object.softmax(attn)
        else:
            pass
            attn = attn_object.softmax(attn)

        ############################
        # B_, num_heads, N, N
        attn_object.q = q#.mean([2,3]).unsqueeze(2)
        attn_object.k = k
        ############################
        logging.debug(f"Swin attn hook: num_heads: {attn_object.num_heads}")
        logging.debug(f"Swin attn hook: window sizes {attn_object.window_size}")
        logging.debug(f"Swin attn hook: head dim {attn_object.head_dim}")
        attn = attn_object.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        return x
    def mhsa_fwd_hook(x):
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(attn_object.kernel_size ** 2)
        if N != num_tokens:
            raise RuntimeError(f"Feature map size ({H} x {W}) is not equal to " +
                               f"expected size ({attn_object.kernel_size} x {attn_object.kernel_size}). " +
                               f"Consider changing sizes or padding inputs.")
        # Faster implementation -- just MHSA
        # If the feature map size is equal to the kernel size, NAT will be equivalent to attn_object-attention.
        qkv = attn_object.qkv(x).reshape(B, N, 3, attn_object.num_heads, attn_object.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * attn_object.scale
        attn = (q @ k.transpose(-2, -1))  # B x heads x N x N
        attn = attn_object.apply_pb(attn)
        attn = attn.softmax(dim=-1)
        ############################
        attn_object.attn_map = attn
        ############################
        attn = attn_object.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        return attn_object.proj_drop(attn_object.proj(x))

    if block_name == "hydraneighborhoodattention":
        return na_fwd_hook
    if block_name == "neighborhoodattentionsplithead":
        return na_legacy_fwd_hook
    elif block_name == "windowattention":
        return swin_fwd_hook
    elif block_name == "mhsarpb":
        return mhsa_fwd_hook
    else:
        raise ValueError(f"Unknown block type {block_name} for attention mask visualization")

def unswin_window(q0, q1, k0, k1, hw, ws, nheads):
    '''
    Attention returns back [B_, nheads, N, C//nheads]
        But N = ws**2
            B_ = B * H/ws * W/ws
    '''
    #print(f"hw is {hw} and ws is {ws}")
    q = torch.concat([q0, q1], dim=1)
    k = torch.concat([k0, k1], dim=1)
    #print(f"q {q.shape}, k {k.shape}")
    q = q.reshape(hw//ws, hw//ws, nheads, ws, ws, -1)
    k = k.reshape(hw//ws, hw//ws, nheads, ws, ws, -1)
    #print(f"q {q.shape}, k {k.shape}")
    q = q.permute(2, 0, 3, 1, 4, 5)
    k = k.permute(2, 0, 3, 1, 4, 5)
    #print(f"q {q.shape}, k {k.shape}")
    q = q.reshape(nheads, hw, hw, -1)
    k = k.reshape(nheads, hw, hw, -1)
    #print(f"q {q.shape}, k {k.shape}")
    q = q.mean(dim=[1,2]).unsqueeze(1)
    k = k.flatten(1,2)
    #print(f"q {q.shape}, k {k.shape}")
    attn = q @ k.transpose(-2, -1)
    #print(f"attn is {attn.shape}")
    attn = attn.reshape(nheads, hw, hw)
    #print(f"attn is {attn.shape}")
    return attn
    


@torch.no_grad()
def visualize_attention(args, generator,
                        name="attn",        # Name prefix for attn map names
                        num_attentions=-1,  # -1 visualizes all
                        cmap='viridis',
                        save_maps=True,
                        log_wandb=False,
                        commit_wandb=False,
                        ):
    if save_maps:
        if "attn_map_path" not in args.evaluation:
            path = args.save_root
        elif args.evaluation.attn_map_path[0] == "/":
            path = args.evaluation.attn_map_path
        else:
            path = args.save_root + args.evaluation.attn_map_path
        if not os.path.exists(path):
            print(f"Path {path} does not exist... making")
            os.mkdir(path)
    # Change generator to have correct hooks
    for i,layer in enumerate(generator.layers):
        logging.debug(f"Layer name {layer.__class__.__name__}")
        for j,block in enumerate(layer.blocks):
            name = block.attn.__class__.__name__
            logging.debug(f"Block name {name}")
            #### SWIN will have module list name
            if name == "ModuleList":
                name = block.attn[0].__class__.__name__
                logging.debug(f"Swin block name {name}")
                for k in len(generator.layers[i].blocks[j].attn):
                    generator.layers[i].blocks[j].attn[k].forward = attn_wrapper(generator.layers[i].blocks[j].attn[k], name)
            # NA or Hydra-NA
            else:
                logging.debug(f"NA block named {block.attn.__class__.__name__}")
                generator.layers[i].blocks[j].attn.forward = attn_wrapper(generator.layers[i].blocks[j].attn, name)

    # Allow us to produce a noise with a constant seed. We can manually set it
    # or just ask it to be constant. We'll save to the arg dict to keep this
    # constant between evaluations and so we can reload if a crash.
    # We only change the rng state for the image sampling, then we set the state
    # back.
    if "const_attn_seed" in args.evaluation and \
            args.evaluation.const_attn_seed is not False:
            _torch_rng_state = torch.random.get_rng_state()
            _py_rng_state = random.getstate()
            # If user used a bool, just set it to 42
            if args.evaluation.const_attn_seed is True:
                with open_dict(args):
                    args.evaluation.attn_seed = 42
            torch.manual_seed(args.evaluation.attn_seed)
            random.seed(args.evaluation.attn_seed)
    noise = torch.randn((1, args.runs.generator.style_dim)).to(args.device)
    sample, _ = generator(noise)
    sample = unnormalize(sample)
    if "const_attn_seed" in args.evaluation and \
            args.evaluation.const_attn_seed is not False:
            torch.set_rng_state(_torch_rng_state)
            random.setstate(_py_rng_state)
    if save_maps:
        save_image(make_grid(sample), f"{path}/original.png")
    if log_wandb:
        wandb.log({'attn_map_original': wandb.Image(make_grid(sample))},
                  commit=False)
    _dict = {}
    nheads_list = [max(c//32, 4) for c in generator.in_channels]
    img_sizes = [2**(i+2) for i in range(9)]
    window_sizes = [2**i if i <=3 else 8 for i in range(2, 11)]
    for i,layer in enumerate(generator.layers):
        if i > 1: # only process above nxn
            for j,block in enumerate(layer.blocks):
                name = block.__class__.__name__
                logging.debug(f"Image Size: {img_sizes[i]} :: name {name}")
                if name == "StyleSwinTransformerBlock":
                    name == "WindowAttention"
                    #window_size = 2**(i+2)
                    q0 = generator.layers[i].blocks[j].attn[0].q
                    q1 = generator.layers[i].blocks[j].attn[1].q
                    k0 = generator.layers[i].blocks[j].attn[0].k
                    k1 = generator.layers[i].blocks[j].attn[1].k
                    attn_map = unswin_window(q0, q1, k0, k1, img_sizes[i], window_sizes[i], nheads_list[i])
                    logging.debug(f"Mean, [min, max] = "\
                            f"{torch.std_mean(attn_map)}, "\
                            f"[{attn_map.min()}, {attn_map.max()}]\n")
                    _dict[f"{name}_{i}{j}"] = attn_map#.softmax(dim=0)
                else:
                    attn_map = generator.layers[i].blocks[j].attn.attn_map#.mean(dim=-1)
                    attn_map /= 2
                    logging.debug(f"Mean, [min, max] = "\
                            f"{torch.std_mean(attn_map)} "\
                            f"[{attn_map.min()}, {attn_map.max()}]\n")
                    attn_map = attn_map.squeeze(0)#.softmax(dim=1)
                    # kernel density channel
                    _dict[f"{name}_{i}{j}"] = attn_map
                    logging.debug(f"Got attn_map with shape {attn_map.shape}")
    # If you want to change the color map play with this
    #_cm = get_cmap(cmap)
    last_key = list(_dict.keys())[-1]
    for k,v in _dict.items():
        logging.debug(f"v: {torch.std_mean(v)} == [{v.min()}, {v.max()}")
        if len(v.shape) == 2:
            nrow = 1
            v = v.unsqueeze(0)
        elif len(v.shape) == 3:
            nrow = math.sqrt(v.shape[0])
            if nrow != int(nrow):
                nrow += 1
            nrow = int(nrow)
            v = v.unsqueeze(1)
        else:
            raise ValueError(f"Got incorrect shape of {v.shape}")
        logging.debug(f"Key {k} with shape {v.shape} :: {nrow}rows:{v.shape[0]%nrow}")
        grid = make_grid(v, nrow=nrow)
        logging.debug(f"Grid shape {grid.shape}")
        if save_maps:
            save_image(grid, f"{path}/{k}.png")
        if log_wandb:
            wandb.log({f"Attn_Map_{k}": wandb.Image(grid)}, commit=False)

    if save_maps:
        print(f"Attention Maps saved to {path}")
