import os
import torch


def compute_iou(pred_mask, real_mask):
    """
        Calculate iou for two masks.
        :param real_mask: tensor of the real mask
        :param pred_mask: tensor of the predicted mask
        :return: IoU score
    """
    real_mask_flat = real_mask.view(-1).bool()
    pred_mask_flat = pred_mask.view(-1).bool()

    intersection = (real_mask_flat & pred_mask_flat).float().sum()  # logical AND
    union = (real_mask_flat | pred_mask_flat).float().sum()

    iou = intersection / union if union != 0 else torch.tensor(1.0)
    return iou


class ConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred_mask, real_mask):
        """
        Update confusion matrix for each batch. Input is a batch of masks.
        :param real_mask: tensor of the real mask
        :param pred_mask: tensor array of the predicted mask
        """
        real_mask_flat = real_mask.view(-1).bool()
        pred_mask_flat = pred_mask.view(-1).bool()

        self.tp += (real_mask_flat & pred_mask_flat).float().sum().item()
        self.fp += (~real_mask_flat & pred_mask_flat).float().sum().item()
        self.fn += (real_mask_flat & ~pred_mask_flat).float().sum().item()
        self.tn += (~real_mask_flat & ~pred_mask_flat).float().sum().item()

    def get_iou(self):
        """
        Calculate iou for each batch.
        :return: IoU score
        """
        iou = self.tp / (self.tp + self.fp + self.fn)
        return iou


# def load_model():
#  waiting for being added...

