import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from models.PortraitNet import PortraitNet
from data.Portraitdataset import EG1800Dataset, SuperviseDataset
from utils.loss_func import TotalLoss
from utils.utils import ConfusionMatrix
import yaml


def predict_one_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        output = torch.sigmoid(output)
        output = output.squeeze()
        output = output.cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        return output
