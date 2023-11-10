import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from data.data_augs import get_texture_transforms, get_deformation_transforms, get_dataset_normalization
import random
import cv2


class EG1800Dataset(Dataset):
    """
    EG1800
    """

    def __init__(self, args, train=True):
        self.train = train
        self.data_root = args.data_root
        self.dataset = args.dataset
        self.img_path = os.path.join(self.data_root, args.dataset, 'Images')
        self.img_list = sorted(os.listdir(self.img_path))
        self.mask_path = os.path.join(self.data_root, args.dataset, 'Labels')
        self.mask_list = sorted(os.listdir(self.mask_path))
        if train:
            self.img_list = self.img_list[:1500]
            self.mask_list = self.mask_list[:1500]
        else:
            self.img_list = self.img_list[1500:1800]
            self.mask_list = self.mask_list[1500:1800]

        self.texture_aug = get_texture_transforms()
        self.deformation_aug = get_deformation_transforms()

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def get_boundary(mask):
        """
        compute boundary using canny
        :param mask:
        :return: canny boundary
        """
        mask = np.array(mask)
        mask[mask > 0] = 1
        boundary = cv2.Canny(mask, threshold1=0, threshold2=1)

        return boundary

    def __getitem__(self, idx):
        img_name = self.img_list[idx].strip()
        mask_name = self.mask_list[idx].strip()
        img = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_path, mask_name)).convert('L')
        img_texture = img.copy()
        if self.train:
            aug = random.random()
            if aug <= 0.3:
                img_texture = self.texture_aug(img)
            elif aug <= 0.5:
                img, mask = self.deformation_aug(img, mask)
        boundary = self.get_boundary(mask)
        img = transforms.ToTensor()(img)
        img_texture = transforms.ToTensor()(img_texture)
        mask = transforms.ToTensor()(mask)  # (0, 1)?
        img = get_dataset_normalization()(img)
        img_texture = get_dataset_normalization()(img_texture)

        return {'Img_name': self.img_list[idx], 'Img_texture': img_texture, 'Img': img,
                'Mask': mask, 'Boundary': boundary}
        # return img, img_texture, mask, boundary
