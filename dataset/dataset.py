from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms as T
from torchvision import datasets


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

def unnormalize(image):
    if image.dim() == 4:
        image[:, 0, :, :] = image[:, 0, :, :] * 0.229 + 0.485
        image[:, 1, :, :] = image[:, 1, :, :] * 0.224 + 0.456
        image[:, 2, :, :] = image[:, 2, :, :] * 0.225 + 0.406
    elif image.dim() == 3:
        image[0, :, :] = image[0, :, :] * 0.229 + 0.485
        image[1, :, :] = image[1, :, :] * 0.224 + 0.456
        image[2, :, :] = image[2, :, :] * 0.225 + 0.406
    else:
        raise NotImplemented(f"Can't handle image of dimension {image.dim()}, please use a 3 or 4 dimensional image")
    return image

def get_dataset(args, evaluation=True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transforms = [T.Resize((args.runs.size, args.runs.size))]
    if args.runs.training.use_flip and not evaluation:
        transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.ToTensor())
    transforms.append(normalize)
    transforms = T.Compose(transforms)

    if args.dataset.lmdb:
        print(f"Using LMDB with {args.dataset.path}")
        dataset = MultiResolutionDataset(path=args.dataset.path,
                                         transform=transforms,
                                         resolution=args.runs.size)
    elif args.dataset.name in ["church"]:
        if args.dataset.name == "church":
            classes = ['church_outdoor_train']
        print(f"Using LSUN {classes}")
        dataset = datasets.LSUN(root=args.dataset.path,
                                transform=transforms,
                                classes=classes)
    elif args.dataset.name in ["cifar10"]:
        print(f"Loading CIFAR-10")
        dataset = datasets.CIFAR10(root=args.dataset.path,
                                   transform=transforms)
    else:
        print(f"Loading ImageFolder dataset from {args.dataset.path}")
        dataset = datasets.ImageFolder(root=args.dataset.path,
                                       transform=transforms)
    return dataset

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return RandomSampler(dataset)
    else:
        return SequentialSampler(dataset)

def get_loader(args, dataset, batch_size=1):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=args.workers,
                        sampler=data_sampler(dataset,
                                             shuffle=True,
                                             distributed=args.distributed),
                        drop_last=True,
                        )
    return loader
