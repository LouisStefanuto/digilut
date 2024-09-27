import torch
import torch.nn as nn


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.05, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss
