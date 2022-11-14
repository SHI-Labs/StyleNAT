import os
import time
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

from dataset.dataset import unnormalize

toPIL = T.ToPILImage()
toTensor = T.ToTensor()

def add_watermark(image, im_size, watermark_path="assets/watermark.jpg",
                  wmsize=16, bbuf=5, opacity=0.9):
    image = image.cpu()
    watermark = Image.open(watermark_path).resize((wmsize, wmsize))
    watermark = toTensor(watermark)
    loc = im_size - wmsize - bbuf
    #image[:,:,loc:-bbuf, loc:-bbuf] = opacity*image[:,:,loc:-bbuf, loc:-bbuf] + watermark
    image[:,:,loc:-bbuf, loc:-bbuf] = watermark
    return image


@torch.no_grad()
def inference(args, generator, dataset):
    assert(os.path.exists(args.inference.save_path)),f"Save path {args.inference.save_path} does not exist"
    for i in range(args.inference.num_images):
        if args.inference.seeds is not None and i < len(args.inference.seeds):
            seed = args.inference.seeds[i]
            torch.random.manual_seed(seed)
        else:
            seed = torch.random.seed()
            print(f"Using Seed: {seed}")

        noise = torch.randn((args.inference.batch, args.runs.generator.style_dim)).to(args.device)
        sample, latent = generator(noise)#, truncation=args.truncation)
        sample = unnormalize(sample)
        sample = add_watermark(sample, im_size=args.runs.size)
        save_image(sample, f"{args.inference.save_path}/{seed}.png",
                   nrow=1, padding=0, normalize=True, value_range=(0,1))
