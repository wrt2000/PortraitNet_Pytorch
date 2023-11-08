import os
import torch


def compute_iou(pred_mask, real_mask):
    """
        Calculate iou for two masks.
        :param real_mask: tensor of the real mask
        :param pred_mask: tensor array of the predicted mask
        :return: IoU score
    """
    real_mask_flat = real_mask.view(-1).bool()
    pred_mask_flat = pred_mask.view(-1).bool()

    intersection = (real_mask_flat & pred_mask_flat).float().sum()  # logical AND
    union = (real_mask_flat | pred_mask_flat).float().sum()

    iou = intersection / union if union != 0 else torch.tensor(1.0)
    return iou


def confusion_matrix():





def load_model():

