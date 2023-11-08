import os
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import random
import torch.nn as nn


class GaussianBlur(object):
    """Apply Gaussian Blur to the input image.
    Args:
       kernel_size (int): Size of the kernel to use.
       sigma (float): Standard deviation of the Gaussian kernel.
    """

    def __init__(self, kernel_size=3, sigma=0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        img = np.array(img)
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, sigma={1})'.format(self.kernel_size, self.sigma)


class GaussianRandomNoise(object):
    """
    Add random gaussian noise to the input image.
    Args:
        mean (float): Mean ("centre") of the Gaussian distribution.
        std (float): Standard deviation (spread or "width") of the Gaussian distribution.
    """

    def __init__(self, mean=0., std=10):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        return img + torch.randn(img.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomSharpness(object):
    """
    Adjust the sharpness of the input image by a random factor.
    Args:
        sharpness_range (float): How much to adjust the sharpness.
    """

    def __init__(self, sharpness_range=(0.8, 1.3)):
        self.sharpness_range = sharpness_range

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        self.sharpness_factor = random.uniform(self.sharpness_range[0], self.sharpness_range[1])  # random choice
        return TF.adjust_sharpness(img, self.sharpness_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(sharpness_factor={0})'.format(self.sharpness_factor)


def get_dataset_normalization():
    return transforms.Normalize([103.94, 116.78, 123.68], [0.017, 0.017, 0.017])


# transforms, need to make sure img and mask are transformed in the same way
# deformation_aug = [
#     # deformation augmentation
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(degrees=(-45, 45)),
#     transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.5)),
#     transforms.RandomAffine(degrees=0, translate=(-0.25, 0.25))
# ]

def get_texture_transforms():
    texture_aug = [
        # texture augmentation, no need to change mask
        GaussianRandomNoise(),  # random noise
        GaussianBlur(kernel_size=random.choice([3, 5])),  # gaussian blur, kernel size 3 or 5 randomly
        transforms.RandomColorJitter(hue=(0.4, 1.7)),  # random color change
        transforms.RandomColorJitter(brightness=(0.4, 1.7)),  # random brightness change
        transforms.RandomColorJitter(contrast=(0.6, 1.5)),  # random contrast change 0.6-1.0
        RandomSharpness(sharpness_range=(0.8, 1.3))  # random sharpness change 0.8-1.3
    ]

    return transforms.RandomChoice(texture_aug)


def get_dataset_denormalize():
    return transforms.Normalize([-103.94, -116.78, -123.68], [1 / 0.017, 1 / 0.017, 1 / 0.017])


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() <= self.p:
            img = TF.hflip(image)
            mask = TF.hflip(mask)
        return img, mask


class RandomRotation:
    def __init__(self, degrees=(-45, 45), p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image, mask):
        if random.random() <= self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            img = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        return img, mask


class RandomResizedCrop:
    def __init__(self, size=(224, 224), scale=(0.5, 1.5), p=0.5):
        self.size = size
        self.scale = scale
        self.p = p

    def __call__(self, image, mask):
        if random.random() <= self.p:
            scale = random.uniform(self.scale[0], self.scale[1])
            img = TF.resized_crop(image, 0, 0, self.size[0], self.size[1], scale=scale)
            mask = TF.resized_crop(mask, 0, 0, self.size[0], self.size[1], scale=scale)
        return img, mask


class RandomAffine:
    def __init__(self, degrees=0, translate=(-0.25, 0.25), p=0.5):
        self.degrees = degrees
        self.translate = translate
        self.p = p

    def __call__(self, image, mask):
        if random.random() <= self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            translate = (
                random.uniform(self.translate[0], self.translate[1]),
                random.uniform(self.translate[0], self.translate[1]))
            img = TF.affine(image, angle=angle, translate=translate, scale=1, shear=0)
            mask = TF.affine(mask, angle=angle, translate=translate, scale=1, shear=0)
        return img, mask


def get_deformation_transforms():
    deformation_aug = [
        # deformation augmentation
        RandomHorizontalFlip(),
        RandomRotation(degrees=(-45, 45)),
        RandomResizedCrop(size=(224, 224), scale=(0.5, 1.5)),
        RandomAffine(degrees=0, translate=(-0.25, 0.25))
    ]

    return transforms.RandomChoice(deformation_aug)
