'''
training for in-distribution classification only
'''

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import models
import datasets
from utils.trainer import train_epoch, eval_epoch

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
parser.add_argument('-d', '--dataset', default='cifar10', choices=dset_names)
parser.add_argument('--scale', default=32, type=int)
parser.add_argument('--crop', default=32, type=int)
parser.add_argument('--no-flip', action='store_true')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-j', '--workers', default=8, type=int)

# optimization
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lr-step', default=30, type=int)
parser.add_argument('--lr-gamma', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)

# evaluation
parser.add_argument('--topk', default=[1, 5], type=int, nargs='+')
parser.add_argument('--print-freq', default=-1, type=int)
parser.add_argument('--test-freq', default=1, type=int)
parser.add_argument('--save-freq', default=10, type=int)


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
if args.dataset in ['mnist', 'svhn'] and not args.no_flip:
    print('Horizontal flip disabled for', args.dataset)
    args.no_flip = True
flip_prob = 0 if args.no_flip else 0.5

pad = args.crop // 8
train_transform = transforms.Compose([
    transforms.Resize(args.scale),
    transforms.RandomCrop(args.crop, padding=pad),
    transforms.RandomHorizontalFlip(flip_prob),
    transforms.ToTensor(),
    normalize
])
val_transform = transforms.Compose([
    transforms.Resize(args.scale),
    transforms.CenterCrop(args.crop),
    transforms.ToTensor(),
    normalize
])

train_dataset, n_class = datasets.__dict__[args.dataset](train=True, transform=train_transform)
val_dataset, _ = datasets.__dict__[args.dataset](train=False, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

# create model
model = models.__dict__[args.arch](n_class)
if args.load_path:
    model.load_state_dict(torch.load(args.load_path))
model = torch.nn.DataParallel(model, args.gpus).cuda()
torch.backends.cudnn.benchmark = True

# optimization
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), args.lr)
else:
    raise ValueError('Invalid optimizer!')
scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)

# start training
for epoch in range(1, args.epochs + 1):
    train_epoch(epoch, train_loader, model, optimizer, scheduler, args)
    if epoch % args.test_freq == 0:
        loss, acc = eval_epoch(epoch, val_loader, model, args)
    if epoch % args.save_freq == 0:
        save_name = args.dataset + '_' + args.arch
        save_path = os.path.join('checkpoints/', save_name + '_{}ep-{:04d}top{}.pth'.format(epoch, round(acc[0] * 10000), args.topk[0]))
        torch.save(model.module.state_dict(), save_path)

for i, k in enumerate(args.topk):
    print('Top {} Accuracy = {:.2%}'.format(k, acc[i]))