import os
import torch


def compute_iou(pred_mask, real_mask):
    """
        Calculate iou for two masks.
        :param real_mask: tensor of the real mask
        :param pred_mask: tensor of the predicted mask
        :return: IoU score
    """
    real_mask = real_mask.view(-1).bool()
    pred_mask = pred_mask.view(-1).bool()

    intersection = (real_mask & pred_mask).float().sum()  # logical AND
    union = (real_mask | pred_mask).float().sum()

    iou = intersection / union if union != 0 else torch.tensor(1.0)
    return iou


class ConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.threshold = 0.5

    def update(self, pred_mask, real_mask):
        """
        Update confusion matrix for each batch. Input is a batch of masks.
        :param real_mask: tensor of the real mask
        :param pred_mask: tensor array of the predicted mask
        """
        threshold = 0.5
        # change 2 channel to 1
        pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
        pred_mask = torch.sigmoid(pred_mask) > self.threshold
        real_mask = torch.sigmoid(real_mask) > self.threshold

        self.tp = self.tp + (real_mask & pred_mask).float().sum().item()
        self.fp = self.fp + (~real_mask & pred_mask).float().sum().item()
        self.fn = self.fn + (real_mask & ~pred_mask).float().sum().item()

    def get_iou(self):
        """
        Calculate iou for each batch.
        :return: IoU score
        """
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-10)
        return iou


# def load_model():
#  waiting ...

