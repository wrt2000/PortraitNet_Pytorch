import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import argparse
import torch.nn as nn
from models.PortraitNet import PortraitNet
from data.Portraitdataset import EG1800Dataset, SuperviseDataset
from utils.loss_func import TotalLoss
from utils.utils import ConfusionMatrix
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import matplotlib.pyplot as plt


def plot_metrics(training_losses, validation_losses, validation_iou, save_dir):
    epochs = range(len(training_losses))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, validation_iou, label='Validation IoU')
    plt.title('Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.show()

def get_logger(args):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File handler
    file_handler = logging.FileHandler(os.path.join(args.save_dir, 'training_log.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='PortraitNet')
    # parser.add_argument('--config', type=str, default='config/supervise.yaml')
    parser.add_argument('--config', type=str, default='config/eg1800.yaml')
    # loss
    parser.add_argument('--Lambda', type=float, default=0.1)  # beta?
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--t', type=float, default=1.0)
    # dataset
    # parser.add_argument('--dataset', type=str, default='Supervisely_face')
    parser.add_argument('--dataset', type=str, default='EG1800')
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--input_size', type=int, default=224)
    # train
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr_decay_epoch', type=int, default=20)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    # save
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--save_freq', type=int, default=500)

    args = parser.parse_args()
    return args


def get_yaml_config(args, config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in config.items():
        if key in args:
            setattr(args, key, value)
    return args


def train(args, train_loader, model, criterion, optimizer, scheduler, epoch, device, logger):
    model.train()
    losses = []
    for i, sample in enumerate(train_loader):
        img = sample['Img'].to(device)
        mask = sample['Mask'].to(device)
        boundary = sample['Boundary'].to(device)
        img_texture = sample['Img_texture'].to(device)
        optimizer.zero_grad()
        mask_def, boundary_def = model(img)
        mask_texture, boundary_texture = model(img_texture)

        loss = criterion(mask_def, mask, boundary_def, boundary, mask_texture, boundary_texture)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # if not?
        if i % 10 == 0:
            logger.info('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
    scheduler.step()

    return sum(losses) / len(losses)


def test(args, val_loader, model, criterion, epoch, device, logger):
    """
    compute validation loss and iou
    """

    model.eval()
    with torch.no_grad():
        losses = []
        confusion_matrix = ConfusionMatrix()
        for i, sample in enumerate(val_loader):
            img = sample['Img'].to(device)
            mask = sample['Mask'].to(device)
            boundary = sample['Boundary'].to(device)
            img_texture = sample['Img_texture'].to(device)
            mask_def, boundary_def = model(img)
            mask_texture, boundary_texture = model(img_texture)
            loss = criterion(mask_def, mask, boundary_def, boundary, mask_texture, boundary_texture)
            losses.append(loss.item())
            confusion_matrix.update(mask_def, mask)

            if i % 10 == 0:
                logger.info('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
        # compute iou
        iou = confusion_matrix.get_iou()
        return sum(losses) / len(losses), iou


def main():
    args = parse_args()
    args = get_yaml_config(args, args.config)
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    os.makedirs(f'{args.save_dir}_{current_time}')
    args.save_dir = f'{args.save_dir}_{current_time}'

    logger = get_logger(args)
    logger.info(args)
    device = torch.device("cuda:0")



    # Dataset
    logger.info('Loading dataset...')
    if args.dataset == 'EG1800':
        train_dataset = EG1800Dataset(args, train=True)
        val_dataset = EG1800Dataset(args, train=False)
    elif args.dataset == 'Supervisely_face':
        train_dataset = SuperviseDataset(args, train=True)
        val_dataset = SuperviseDataset(args, train=False)
    else:
        raise NotImplementedError
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info('Finish loading dataset! Total training examples: {}, Total validation examples: {}'.format(
        train_dataset.__len__(), val_dataset.__len__()))

    # Model
    logger.info('Building model...')
    model = PortraitNet().to(device)
    logger.info(model)
    logger.info('Finish building model!')

    # Loss
    criterion = TotalLoss(args).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)

    # Train
    logger.info('Start training...')
    training_losses = []
    validation_losses = []
    for epoch in range(args.epochs):
        logger.info('Epoch: {}'.format(epoch))
        train_loss = train(args, train_loader, model, criterion, optimizer, scheduler, epoch, device, logger)
        test_loss, test_iou = test(args, val_loader, model, criterion, epoch, device, logger)
        training_losses.append(train_loss)
        validation_losses.append(test_loss)

        if (epoch + 1) % args.save_freq == 0:
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'validation_iou': test_iou
            }
            torch.save(ckpt_dict, os.path.join(args.save_dir, 'ckpt_epoch_{}.pth'.format(epoch)))
        logger.info('Epoch: {}, Train Loss: {}, Validation Loss: {}, Validation IoU: {}'.format(
            epoch, train_loss, test_loss, test_iou))


if __name__ == '__main__':
    main()
