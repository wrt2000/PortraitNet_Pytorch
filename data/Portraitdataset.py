import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from data.data_augs import get_texture_transforms, get_deformation_transforms, get_dataset_normalization
import random
import cv2
import glob


class EG1800Dataset(Dataset):
    """
    EG1800
    The number of images is smaller than the number of masks!!!
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
        boundary = cv2.Canny(np.uint8(mask), threshold1=0.3, threshold2=0.5)

        return boundary

    def __getitem__(self, idx):
        img_name = self.img_list[idx].strip()
        img = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_path, img_name)).convert('L')
        img = img.resize((224, 224), Image.BILINEAR)
        mask = mask.resize((224, 224), Image.BILINEAR)
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
        mask = transforms.ToTensor()(mask) * 255.0  # (0, 1)?
        # img = get_dataset_normalization()(img)
        # img_texture = get_dataset_normalization()(img_texture)
        boundary = torch.from_numpy(boundary / 255.0).unsqueeze(0)

        return {'Img_name': self.img_list[idx], 'Img_texture': img_texture.to(torch.float32),
                'Img': img.to(torch.float32), 'Mask': mask.to(torch.float32), 'Boundary': boundary.to(torch.float32)}
        # return img, img_texture, mask, boundary


class SuperviseDataset(EG1800Dataset):
    def __init__(self, args, train=True):
        self.train = train
        self.data_root = args.data_root
        self.dataset = args.dataset
        self.img_list = sorted(glob.glob(os.path.join(self.data_root, args.dataset, '*/resize_img', '*.png')))
        self.mask_list = sorted(glob.glob(os.path.join(self.data_root, args.dataset, '*/resize_mask', '*.png')))
        len_dataset = int(len(self.img_list) * 0.8)
        if train:
            self.img_list = self.img_list[:len_dataset]
            self.mask_list = self.mask_list[:len_dataset]
        else:
            self.img_list = self.img_list[len_dataset:]
            self.mask_list = self.mask_list[len_dataset:]

        self.texture_aug = get_texture_transforms()
        self.deformation_aug = get_deformation_transforms()

    def __getitem__(self, idx):
        img_name = self.img_list[idx].strip()
        mask_name = img_name.split('/')[-1]
        mask_name = glob.glob(os.path.join(self.data_root, self.dataset, '*/resize_mask', mask_name))[0]
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        img = img.resize((224, 224), Image.BILINEAR)
        mask = mask.resize((224, 224), Image.BILINEAR)
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
        boundary = torch.from_numpy(boundary / 255.0).unsqueeze(0)

        return {'Img_name': self.img_list[idx], 'Img_texture': img_texture.to(torch.float32),
                'Img': img.to(torch.float32), 'Mask': mask.to(torch.float32), 'Boundary': boundary.to(torch.float32)}



