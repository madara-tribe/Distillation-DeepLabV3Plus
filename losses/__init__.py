import torch.nn as nn
from .focal_loss import FocalLoss
from .kldiv_loss import KLDivergenceLoss

class DistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kldiv_loss = KLDivergenceLoss()
        
    def forward(self, preds, labels):
        kldiv_loss = self.kldiv_loss(preds, labels)
        return kldiv_loss
        
class MultiClassCriterion(nn.Module):
    def __init__(self, loss_type='Lovasz', **kwargs):
        super().__init__()
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss

