import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

"""
Total loss consists of:
    1. texture aug loss of mask and boundary (focal loss)
    2. deformation aug loss of mask and boundary (focal loss)
    3. consistency constraint loss
    
    L = L_m + α * Lc + β * Le

"""


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduce=True):
        """
        :param gamma:
        :param reduce: if true, return the mean of all examples' loss
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        """
        :param inputs: shape (N, C, H, W)
        :param targets: shape (N, H, W)
        :return: KL loss
        """
        inputs = inputs.view(-1, inputs.size(-1))  # (N*H*W, C), C means classes, in segmentation, C=2
        targets = targets.view(-1, 1)  # (N*H*W, 1)
        logpt = F.log_softmax(inputs, dim=-1)  # (N*H*W, C)
        logpt = logpt.gather(1, targets)  # (N*H*W, 1)  gather the logpt of the target class
        logpt = logpt.view(-1)  # (N*H*W)
        pt = torch.exp(logpt)
        loss = -1 * ((1 - pt) ** self.gamma * logpt + pt ** self.gamma * torch.log(1 - pt))  # (N*H*W)
        if self.reduce:
            return loss.mean()
        else:
            return loss


# consistency constraint loss
class KLLoss(nn.Module):
    def __init__(self, t=1.0):
        super(KLLoss, self).__init__()
        self.t = t

    def forward(self, inputs, targets):
        """
        heatmap
        :param inputs: shape (N, C, H, W)
        :param targets: shape (N, C, H, W)
        :return: KL loss
        """
        inputs = inputs.view(-1, inputs.size(-1))  # (N*H*W, C)
        targets = targets.view(-1, targets.size(-1))  # (N*H*W, C)
        inputs = F.softmax(inputs / self.t, dim=-1)
        targets = F.softmax(targets / self.t, dim=-1)
        loss = F.kl_div(inputs, targets, reduction='batchmean')  # (N*H*W), batchmean: sum up the loss of all examples
        return loss


class TotalLoss(nn.Module):
    def __init__(self, args):
        super(TotalLoss, self).__init__()
        self.args = args
        self.focal_loss = FocalLoss(gamma=args.gamma)
        self.kl_loss = KLLoss(t=args.t)
        self.bce_loss = nn.BCELoss()

    def forward(self, mask_deformation, mask, boundary_deformation, boundary, mask_texture, boundary_texture):
        """
        :param mask_deformation: shape (N, C, H, W)
        :param mask: shape (N, C, H, W)
        :param boundary_deformation: shape (N, C, H, W)
        :param boundary: shape (N, C, H, W)
        :param mask_texture: shape (N, C, H, W)
        :param boundary_texture: shape (N, C, H, W)
        :return: total loss
        """
        # texture loss
        loss_mask_texture = self.bce_loss(mask_texture, mask)
        loss_boundary_texture = self.focal_loss(boundary_texture, boundary)

        # deformation loss
        loss_mask_deformation = self.bce_loss(mask_deformation, mask)
        loss_boundary_deformation = self.focal_loss(boundary_deformation, boundary)

        # consistency constraint loss
        loss_mask_consistency = self.kl_loss(mask_deformation, mask_texture)
        loss_boundary_consistency = self.kl_loss(boundary_deformation, boundary_texture)

        # total loss
        loss_mask = loss_mask_texture + loss_mask_deformation + self.args.alpha * loss_mask_consistency
        loss_boundary = self.args.Lambda * (loss_boundary_texture + loss_boundary_deformation) + \
                        self.args.alpha * loss_boundary_consistency

        return loss_mask + loss_boundary
