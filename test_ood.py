'''
OOD detection performance
'''

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import models
import datasets
from utils.meters import auroc, aupr, fpr_tpr

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))

sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default=[], type=int, nargs='+')

# model info
parser.add_argument('-a', '--arch', default='wrn40', choices=model_names)
parser.add_argument('--load-path', type=str)

# dataset info
parser.add_argument('-id', '--in-dataset', default='cifar10', choices=dset_names)
parser.add_argument('-od', '--out-dataset', default='svhn', choices=dset_names, nargs='+')
parser.add_argument('--scale', default=32, type=int)
parser.add_argument('--crop', default=32, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-j', '--workers', default=4, type=int)

# settings
parser.add_argument('--repeat', default=5, type=int)
parser.add_argument('--out-ratio', default=0.2, type=float)

def conf_scores(model, loader, progress=False):
    model.eval()
    all_scores = []
    if progress:
        loader = tqdm(loader)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(args.gpus[0]), targets.cuda(args.gpus[0])
            outputs = model(inputs)
            scores = F.softmax(outputs, 1).max(1)[0]
            all_scores.append(scores)
    return torch.cat(all_scores, 0)

# get available gpus
args = parser.parse_args()
if args.gpus[0] < 0:
    import GPUtil
    n_gpus = -args.gpus[0] if args.gpus[0] < -1 else 4
    args.gpus = [int(i) for i in GPUtil.getAvailable(order='first', limit=n_gpus, maxMemory=0.15)]
    if len(args.gpus) < n_gpus:
        raise RuntimeError('No enough GPUs')
    print('Using GPUs:', *args.gpus)
torch.cuda.set_device(args.gpus[0])

# data loading
normalize = transforms.Normalize(mean=[0.5] * 3, std=[0.25] * 3)

transform = transforms.Compose([
    transforms.Resize(args.scale),
    transforms.CenterCrop(args.crop),
    transforms.ToTensor(),
    normalize
])

in_dataset, n_class = datasets.__dict__[args.in_dataset](train=False, transform=transform)
in_loader = DataLoader(in_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

out_num = int(args.out_ratio * len(in_dataset))

# create model
model = models.__dict__[args.arch](n_class)
if args.load_path:
    model.load_state_dict(torch.load(args.load_path, map_location=lambda storage, loc: storage))
model = torch.nn.DataParallel(model, args.gpus).cuda()
in_scores = conf_scores(model, in_loader, progress=True)

all_fprs = []
all_aurocs = []
all_auprs = []

for d in args.out_dataset:
    out_dataset, _ = datasets.__dict__[d](train=False, transform=transform)
    print('OOD dataset {}: {:d} examples'.format(d, len(out_dataset)))
    assert len(out_dataset) >= out_num

    fprs = []
    aurocs = []
    auprs = []
    for _ in range(args.repeat):
        out_ind = np.random.choice(len(out_dataset), out_num, replace=False)
        out_dataset_sample = torch.utils.data.Subset(out_dataset, out_ind)
        out_loader = DataLoader(out_dataset_sample, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        # compute ood performance
        out_scores = conf_scores(model, out_loader)
        fprs.append(100 * fpr_tpr(in_scores, out_scores, 0.95))
        aurocs.append(100 * auroc(in_scores, out_scores))
        auprs.append(100 * aupr(in_scores, out_scores))

    print('FPR95:\t{:.2f} ({:.2f})'.format(np.mean(fprs), np.std(fprs)))
    print('AUROC:\t{:.2f} ({:.2f})'.format(np.mean(aurocs), np.std(aurocs)))
    print('AUPR:\t{:.2f} ({:.2f})'.format(np.mean(auprs), np.std(auprs)))

    all_fprs += fprs
    all_aurocs += aurocs
    all_auprs += auprs
    
print('_____________________')
print('All OOD datasets')
print('FPR95:\t{:.2f}'.format(np.mean(all_fprs)))
print('AUROC:\t{:.2f}'.format(np.mean(all_aurocs)))
print('AUPR:\t{:.2f}'.format(np.mean(all_auprs)))