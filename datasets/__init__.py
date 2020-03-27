import os
import torch
from torchvision import datasets
root_dir = '/path/to/root/'

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


def cifar10(train, transform):
    dataset = datasets.CIFAR10(root_dir, train=train, transform=transform, download=True)
    return dataset, 10

def cifar100(train, transform):
    dataset = datasets.CIFAR100(root_dir, train=train, transform=transform, download=True)
    return dataset, 100

def svhn(train, transform):
    split = 'train' if train else 'test'
    dataset = datasets.SVHN(root_dir, split=split, transform=transform, download=True)
    return dataset, 10

def lsun(train, transform):
    split = 'train' if train else 'test'
    dataset = datasets.LSUN(root_dir + 'lsun', classes=split, transform=transform)
    return dataset, None

def places(train, transform):
    # dataset = datasets.ImageFolder('/path/to/places/', transform=transform)
    dataset = datasets.ImageFolder('/data7/yili/imgDB/places/', transform=transform)
    return dataset, len(dataset.classes)

def textures(train, transform):
    dataset = datasets.ImageFolder('/path/to/dtd/images/', transform=transform)
    return dataset, len(dataset.classes)

def tiny_images(train, transform):
    from .tinyimages import TinyImages
    print('WARNING: train={} ignored'.format(train))
    dataset = TinyImages('/path/to/tiny-images/', transform=transform, exclude_cifar=True)
    return dataset, None

def tiny_imagenet(train, transform):
    from .tinyimagenet import TinyImageNet
    split = 'train' if train else 'val'
    dataset = TinyImageNet('/path/to/tiny-imagenet-200/', split=split, transform=transform)
    return dataset, 200

def ilsvrc(train, transform):
    from .folder2lmdb import ImageFolderLMDB
    split = 'train' if train else 'val'
    db_path = '/path/to/ILSVRC12/%s.lmdb' % split
    dataset = ImageFolderLMDB(db_path, transform=transform)
    return dataset, 1000

def gaussian(train, transform, sigma=0.5, n=10000):
    targets_tensor = torch.zeros(n)
    inputs_tensor = sigma * torch.randn(n, 3, 32, 32)
    inputs_tensor.clamp_(-1, 1)
    dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
    return dataset, None

def uniform(train, transform, xmin=-1, xmax=1, n=10000):
    targets_tensor = torch.zeros(n)
    inputs_tensor = xmin + (xmax - xmin) * torch.rand(n, 3, 32, 32)
    dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
    return dataset, None