import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets, topk=[1]):
    N = outputs.size(0)
    pred = outputs.topk(max(topk), dim=1)[1]
    correct = pred.t().eq(targets).float()
    accs = [correct[:k].sum().item() / N for k in topk]
    return accs


def auroc(in_scores, out_scores, plot=False):
    '''
    calculate the area under precision-recall curve
    '''
    num_in = len(in_scores)
    num_out = len(out_scores)
    y_true = np.concatenate([np.zeros(num_in), np.ones(num_out)])
    y_score = np.concatenate([in_scores.detach().cpu().numpy(), out_scores.detach().cpu().numpy()])
    auc = roc_auc_score(y_true, 1 - y_score)
    if plot:
        fpr, tpr, _ = roc_curve(y_true, 1 - y_score)
        return auc, fpr, tpr
    return auc


def aupr(in_scores, out_scores, plot=False):
    '''
    calculate the area under precision-recall curve
    '''
    num_in = len(in_scores)
    num_out = len(out_scores)
    y_true = np.concatenate([np.zeros(num_in), np.ones(num_out)])
    y_score = np.concatenate([in_scores.detach().cpu().numpy(), out_scores.detach().cpu().numpy()])
    auc = average_precision_score(y_true, 1 - y_score)
    if plot:
        prec, rec, _ = precision_recall_curve(y_true, 1 - y_score)
        return auc, prec, rec
    return auc


def fpr_tpr(in_scores, out_scores, tpr_level=0.95, eps=0.01):
    '''
    calculate the false positive error rate when tpr is 95%
    '''
    in_scores = in_scores.detach().cpu().numpy()
    out_scores = out_scores.detach().cpu().numpy()

    thresh = np.quantile(out_scores, tpr_level)
    fpr = np.mean(in_scores <= thresh)

    return fpr