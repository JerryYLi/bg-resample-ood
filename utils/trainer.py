import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .meters import AverageMeter, accuracy

def train_epoch(epoch, loader, model, optimizer, scheduler, args):
    model.train()

    losses = AverageMeter()
    accs = [AverageMeter() for _ in args.topk]
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(args.gpus[0]), targets.cuda(args.gpus[0])

        # forward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))

        # evaluate
        acc = accuracy(outputs, targets, args.topk)
        for j in range(len(accs)):
            accs[j].update(acc[j], inputs.size(0))
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.print_freq > 0 and (i + 1) % args.print_freq == 0:
            print('Epoch {}\tIteration {}/{}\tLoss {:.3f}\tPrec {:.2%}@{:d}\t{:.2%}@{:d}'
                  .format(epoch, i + 1, len(loader), losses.val, acc[0], args.topk[0], acc[1], args.topk[1]))

    if scheduler is not None:
        scheduler.step(epoch)

    print('Epoch {:d}\tTrain loss {:.3f}'.format(epoch, losses.avg))
    avg_accs = []
    for j in range(len(accs)):
        acc = accs[j].avg
        avg_accs.append(acc)
        print('Train prec@{} {:.2%}'.format(args.topk[j], acc))
    return losses.avg, avg_accs


def eval_epoch(epoch, loader, model, args):
    model.eval()

    losses = AverageMeter()
    accs = [AverageMeter() for _ in args.topk]
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(loader)):
            inputs, targets = inputs.cuda(args.gpus[0]), targets.cuda(args.gpus[0])

            # forward
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            # evaluate
            acc = accuracy(outputs, targets, args.topk)
            for j in range(len(accs)):
                accs[j].update(acc[j], inputs.size(0))
    
    print('Epoch {:d}\tVal loss {:.3f}'.format(epoch, losses.avg))
    avg_accs = []
    for j in range(len(accs)):
        acc = accs[j].avg
        avg_accs.append(acc)
        print('Val prec@{} {:.2%}'.format(args.topk[j], acc))
    return losses.avg, avg_accs


def train_epoch_dual(epoch, loader1, loader2, model, loss2, optimizer, scheduler, args):
    model.eval()

    losses = AverageMeter()
    accs = [AverageMeter() for _ in args.topk]
    for i, (inputs, targets) in enumerate(loader1):
        N = inputs.size(0)
        inputs2, _ = next(loader2)
        inputs = torch.cat((inputs, inputs2), 0)
        inputs, targets = inputs.cuda(args.gpus[0]), targets.cuda(args.gpus[0])
        
        # forward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs[:N], targets) + loss2(outputs[N:])
        losses.update(loss.item(), inputs.size(0))

        # evaluate
        acc = accuracy(outputs[:N], targets, args.topk)
        for j in range(len(accs)):
            accs[j].update(acc[j], inputs.size(0))
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.print_freq > 0 and (i + 1) % args.print_freq == 0:
            print('Epoch {}\tIteration {}/{}\tLoss {:.3f}\tPrec {:.2%}@{:d}\t{:.2%}@{:d}'
                  .format(epoch, i + 1, len(loader), losses.val, acc[0], args.topk[0], acc[1], args.topk[1]))

    if scheduler is not None:
        scheduler.step(epoch)

    print('Epoch {:d}\tTrain loss {:.3f}'.format(epoch, losses.avg))
    avg_accs = []
    for j in range(len(accs)):
        acc = accs[j].avg
        avg_accs.append(acc)
        print('Train prec@{} {:.2%}'.format(args.topk[j], acc))
    return losses.avg, avg_accs


def eval_epoch_dual(epoch, loader1, loader2, model, loss2, args):
    model.eval()

    losses = AverageMeter()
    accs = [AverageMeter() for _ in args.topk]
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(loader1)):
            N = inputs.size(0)
            inputs2, _ = next(loader2)
            inputs = torch.cat((inputs, inputs2), 0)
            inputs, targets = inputs.cuda(args.gpus[0]), targets.cuda(args.gpus[0])
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs[:N], targets) + loss2(outputs[N:])
            losses.update(loss.item(), inputs.size(0))
            
            acc = accuracy(outputs[:N], targets, args.topk)
            for j in range(len(accs)):
                accs[j].update(acc[j], inputs.size(0))
    
    print('Epoch {:d}\tVal loss {:.3f}'.format(epoch, losses.avg))
    avg_accs = []
    for j in range(len(accs)):
        acc = accs[j].avg
        avg_accs.append(acc)
        print('Val prec@{} {:.2%}'.format(args.topk[j], acc))
    return losses.avg, avg_accs


def train_epoch_resample(epoch, loader1, loader2, weight_params, model, loss2, optimizer, optimizer_w, scheduler, args):
    model.eval()

    losses = AverageMeter()
    losses_w = AverageMeter()
    accs = [AverageMeter() for _ in args.topk]
    for i, (inputs, targets) in enumerate(loader1):
        N = inputs.size(0)
        indices, inputs2, _ = next(loader2)
        inputs = torch.cat((inputs, inputs2), 0)
        inputs, targets = inputs.cuda(args.gpus[0]), targets.cuda(args.gpus[0])

        # resampling weights
        w = F.softplus(weight_params)
        scale = (w ** .5).mean() / (w.mean() ** .5)
        weights = scale * (w[indices.cuda(args.gpus[0])] / w.mean()) ** .5
        
        # forward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs[:N], targets)
        loss_ood = loss2(outputs[N:], weights)
        loss += loss_ood
        losses.update(loss.item(), inputs.size(0))
        loss_w = -loss_ood
        losses_w.update(loss_w.item(), inputs.size(0))
        
        # evaluate
        acc = accuracy(outputs[:N], targets, args.topk)
        for j in range(len(accs)):
            accs[j].update(acc[j], inputs.size(0))
        
        # backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer_w.zero_grad()
        loss_w.backward()
        optimizer_w.step()

        if args.print_freq > 0 and (i + 1) % args.print_freq == 0:
            print('Epoch {}\tIteration {}/{}\tLoss {:.3f}\tLoss_w {:.3f}\tPrec {:.2%}@{:d}\t{:.2%}@{:d}'
                  .format(epoch, i + 1, len(loader1), losses.val, losses_w.val, acc[0], args.topk[0], acc[1], args.topk[1]))

    if scheduler is not None:
        scheduler.step(epoch)

    print('w min={:.3f}, avg={:.3f}, max={:.3f}'.format(w.min(), w.mean(), w.max()))
    print('Epoch {:d}\tTrain loss {:.3f}\tloss_w = {:.3f}'.format(epoch, losses.avg, losses_w.avg))
    avg_accs = []
    for j in range(len(accs)):
        acc = accs[j].avg
        avg_accs.append(acc)
        print('Train prec@{} {:.2%}'.format(args.topk[j], acc))
    return losses.avg, avg_accs