import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


eps = 1e-3
IDX_CYST = 1


def dice_loss(preds, trues, weight=None, is_average=True, cyst=False):
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    #if cyst:
    #    intersection = (preds * trues)
    #    inter = (intersection / (intersection + 1e-5)).sum(1)
    #    intersection = intersection.sum(1)
    #    scores = (
    #        (2. * intersection + eps) 
    #        / (preds.sum(1) + .1 * (trues.sum(1) - inter) + inter + eps))
    #else:
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    if is_average:
        score = scores.sum() / num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


def jaccard(preds, trues, weight=None, is_average=True, cyst=False):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w

   # if cyst:
   #     intersection = (preds * trues).sum(1)
   #     scores = (intersection + eps) / (preds.sum(1) + .1 * trues.sum(1) - intersection + eps)
   # else:
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    if is_average:
        score = scores.sum() / num
        return torch.clamp(score, 0., 1.)
    else:
        return scores
    
def bce_for_chnl(preds, trues, is_average=True, cyst=False):
    num = preds.size(0)
    scores = F.binary_cross_entropy_with_logits(
        preds, trues, reduce=is_average
    )
    if not is_average:
        return scores.view(num, -1).mean(1)
    else:
        return scores


def per_class_loss(loss, preds, trues, ch_weights):
    loss_meter = []
    for idx in range(preds.shape[1]):
        loss_meter.append(
            ch_weights[:, idx] * loss(
                preds[:,idx,...].contiguous(), 
                trues[:,idx,...].contiguous(),
                is_average=False,
                cyst=(idx == IDX_CYST)
            ))
    return loss_meter


def semantic_loss(loss, preds, trues, ch_weights):
    channels = per_class_loss(loss, preds, trues, ch_weights)
    return sum(channels).sum() / ch_weights.sum()


class BCEDiceJaccardLoss(nn.Module):
    def __init__(self, weights, size_average=True, chnls_w=None):
        super().__init__()
        self.weights = weights
        self.chnls_w = chnls_w
        if self.chnls_w is None:
            self.chnls_w = np.ones((1, 6))
        self.chnls_w = torch.Tensor(self.chnls_w).cuda()
        self.bce = bce_for_chnl
        self.jacc = jaccard
        self.dice = dice_loss
        self.mapping = {'bce': self.bce,
                        'jacc': self.jacc,
                        'dice': self.dice}
        self.values = {}

    def forward(self, preds, target):
        loss = 0
        num = target.size(0)
        chs = target.size(1)
        ch_weights = (target.view(num, chs, -1).sum(2) > 0).float()
        ch_weights *= self.chnls_w
        for k in ['bce', 'jacc', 'dice']:
            if not self.weights[k]:
                continue
            
            val = semantic_loss(self.mapping[k], preds, target, ch_weights)
            self.values[k] = val

            if k != 'bce':
                loss += self.weights[k] * (1 - val)
            else:
                loss += self.weights[k] * val
                preds = F.sigmoid(preds)

        return loss


#def dice_clamp(preds, trues, is_average=True):
#    preds = torch.round(preds)
#    return dice_loss(preds, trues, is_average=is_average)
#
#
#class DiceLoss(nn.Module):
#    def __init__(self, size_average=True):
#        super().__init__()
#        self.size_average = size_average
#
#    def forward(self, input, target, weight=None):
#        return 1 - dice_loss(F.sigmoid(input), target, weight=weight, is_average=self.size_average)
#
#    
#class BCEDiceLoss(nn.Module):
#    def __init__(self, dice_coef=0.4, size_average=True):
#        super().__init__()
#        self.dice_coef = dice_coef
#        self.size_average = size_average
#        self.dice = DiceLoss(size_average=size_average)
#
#    def forward(self, input, target, weight=None):
#        return ((1 - self.dice_coef) * nn.BCEWithLogitsLoss(
#            size_average=self.size_average, weight=weight)(input, target) + 
#                self.dice_coef * self.dice(input, target, weight=weight))
#
#
#def soft_jaccard(outputs, targets):
#    eps = 1e-15
#    jaccard_target = (targets == 1).float()
#    jaccard_output = F.sigmoid(outputs)
#
#    intersection = (jaccard_output * jaccard_target).sum()
#    union = jaccard_output.sum() + jaccard_target.sum()
#    return intersection / (union - intersection + eps)
#
#
#class LossBinary:
#    """
#    Loss defined as BCE - log(soft_jaccard)
#    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
#    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
#    arXiv:1706.06169
#    """
#
#    def __init__(self, jaccard_weight=0):
#        self.nll_loss = nn.BCEWithLogitsLoss()
#        self.jaccard_weight = jaccard_weight
#
#    def __call__(self, outputs, targets):
#        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
#
#        if self.jaccard_weight:
#            loss += self.jaccard_weight * (1 - soft_jaccard(outputs, targets))
#        return loss
