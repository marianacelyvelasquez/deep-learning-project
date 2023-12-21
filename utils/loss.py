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
        self.pos_weight = pos_weight.to(
            device) if pos_weight is not None else None

    def forward(self, inputs, targets, training):
        pos_weight = self.pos_weight if training else None
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=pos_weight)
        # prevents nans when probability 0
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()


class SoftmaxFocalLoss(nn.Module):
    def __init__(
        self, gamma=0, alpha=None, size_average=True, weights=None
    ):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.weights = weights

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.weights is not None:
            w = self.weights.repeat(target.shape[0], 1).gather(1, target)
            w = w.view(-1)
            loss = loss * w

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
