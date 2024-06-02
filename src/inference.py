import os
import time
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

from dataset.dataset import unnormalize

toPIL = T.ToPILImage()
toTensor = T.ToTensor()

@torch.inference_mode()
def add_watermark(image, im_size, watermark_path="assets/watermark.jpg",
                  wmsize=16, bbuf=5, opacity=0.9):
    '''
    Creates a watermark on the saved inference image.
    We request that you do not remove this to properly assign credit to
    Shi-Lab's work.
    '''
    image = image.cpu()
    watermark = Image.open(watermark_path).resize((wmsize, wmsize))
    watermark = toTensor(watermark)
    loc = im_size - wmsize - bbuf
    image[:,:,loc:-bbuf, loc:-bbuf] = watermark
    return image

def extract_range(args):
    '''
    Helper function to allow for a range of inputs from hydra-core
    This only works if you specify a range from the config file. 
        seeds: range(start, end, step)

    If you are using the command line pass in a comma delimited sequence with
    `seq`
        python main type=inference inference.seeds=[`seq -s, start step_size end`]
    The back ticks are important because they run a bash command
    '''
    start = 0
    step_size=1
    start_end = args.inference.seeds.split("range(")[1].split(")")[0]
    if "," in start_end:
        nums = start_end.split(',')
        match len(nums):
            case 1:
                end = nums[0]
            case 2:
                start = nums[0]
                end = nums[1]
            case 3:
                start = nums[0]
                end = nums[1]
                step_size = nums[2]
            case _:
                raise ValueError
    else:
        end = start_end
    args.inference.seeds = list(range(int(start), int(end), int(step_size)))
    print(f"Using Range of Seeds from {start} to {end} with a step size of {step_size}")

@torch.inference_mode()
def inference(args, generator):
    save_path = args.inference.save_path
    if save_path[0] != "/":
        save_path = args.save_root + save_path
    assert(os.path.exists(save_path)),f"Path {save_path} does not exist"
    assert('num_images' in args.inference or 'seeds' in args.inference),\
            f"Inference must either specify a number of images "\
            f"(random seed generation) or a set of seeds to use to generate."
    if 'num_images' in args.inference:
        num_imgs = args.inference.num_images
    if "seeds" in args.inference and args.inference.seeds is not None: 
        # Handles "range(start, end)" input from hydra file 
        if "range" in args.inference.seeds:
            extract_range(args)
            num_imgs = len(args.inference.seeds)
        else:
            num_imgs = len(args.inference.seeds)

    for i in range(num_imgs):
        if "seeds" in args.inference and args.inference.seeds is not None and \
                i < len(args.inference.seeds):
            seed = args.inference.seeds[i]
            torch.random.manual_seed(seed)
        else:
            seed = torch.random.seed()
            print(f"Using Seed: {seed}")

        noise = torch.randn((args.inference.batch,
                             args.runs.generator.style_dim)).to(args.device)
        sample, latent = generator(noise)
        sample = unnormalize(sample)
        sample = add_watermark(sample, im_size=args.runs.size)
        save_image(sample, f"{save_path}/{seed}.png",
                   nrow=1, padding=0, normalize=True, value_range=(0,1))
