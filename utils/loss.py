import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BinaryFocalLoss(nn.Module):
    def __init__(self, device, gamma=2, alpha=1, pos_weight=None):
        """"""
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight.to(device) if pos_weight is not None else None

    def forward(self, inputs, targets, training):
        pos_weight = self.pos_weight if training else None
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", pos_weight=pos_weight
        )
        # prevents nans when probability 0
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
