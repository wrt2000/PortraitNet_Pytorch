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
        :return: focal loss
        """
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, targets.size(-1)).to(torch.int64)
        logpt = F.log_softmax(inputs, dim=-1)
        logpt = logpt.gather(1, targets)  # get the log of the ground truth class
        logpt = logpt.view(-1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduce:
            loss = loss.mean()

        assert loss >= 0, "Focal loss is negative."
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
        # inputs = inputs.view(-1, inputs.size(-1))  # (N*H*W, C)
        # targets = targets.view(-1, targets.size(-1))  # (N*H*W, C)
        # log_softmax: log of softmax
        inputs = F.log_softmax(inputs / self.t, dim=1)
        # make sure inputs is not inf and not contain nan
        assert torch.isfinite(inputs).all(), f"inputs is not finite. {inputs}"
        targets = F.softmax(targets / self.t, dim=1)
        assert torch.isfinite(targets).all(), f"targets is not finite. {targets}"
        # (N*H*W), batchmean: sum up the loss of all examples
        loss = F.kl_div(inputs, targets, reduction='batchmean') * self.t * self.t  # input: log
        # make sure the loss is non-negative
        assert loss >= 0, f"KL loss is negative. {loss}"
        return loss


class TotalLoss(nn.Module):
    def __init__(self, args):
        super(TotalLoss, self).__init__()
        self.args = args
        self.focal_loss = FocalLoss(gamma=args.gamma)
        self.kl_loss = KLLoss(t=args.t)
        self.ce_loss = nn.CrossEntropyLoss()

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
        loss_mask_texture = self.ce_loss(mask_texture, mask)
        loss_boundary_texture = self.focal_loss(boundary_texture, boundary)

        # deformation loss
        loss_mask_deformation = self.ce_loss(mask_deformation, mask)
        loss_boundary_deformation = self.focal_loss(boundary_deformation, boundary)

        # consistency constraint loss
        loss_mask_consistency = self.kl_loss(mask_deformation, mask_texture)
        loss_boundary_consistency = self.kl_loss(boundary_deformation, boundary_texture)

        # total loss
        loss_mask = loss_mask_texture + loss_mask_deformation + self.args.alpha * loss_mask_consistency
        loss_boundary = self.args.Lambda * (loss_boundary_texture + loss_boundary_deformation) + \
                        self.args.alpha * loss_boundary_consistency
        assert loss_mask + loss_boundary >= 0, "Total loss is negative."

        return loss_mask + loss_boundary
