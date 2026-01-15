import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import time
import cv2
from sklearn.model_selection import KFold
import yaml
from easydict import EasyDict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

import pandas as pd

warnings.filterwarnings('ignore')


class Dataset(data.Dataset):
    def __init__(self, image_root, gt_root, augmentations, sample):
        self.image_root = image_root
        self.gt_root = gt_root

        self.samples = sample
        self.transform = augmentations

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.image_root + '/' + name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        mask = cv2.imread(self.gt_root + '/' + name, cv2.IMREAD_GRAYSCALE) / 255.0
        pair = self.transform(image=image, mask=mask)

        return pair['image'], pair['mask'].unsqueeze(0), int(name.replace('.png', ''))

    def __len__(self):
        return len(self.samples)


class Multi_Center_Dataset(data.Dataset):
    def __init__(self, image_root, gt_root, augmentations):
        self.image_root = image_root
        self.gt_root = gt_root

        self.samples = [name for name in os.listdir(image_root) if name[0] != "."]

        self.transform = augmentations


    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.image_root + '/' + name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


        mask = cv2.imread(self.gt_root + '/' + name, cv2.IMREAD_GRAYSCALE) / 255.0
        pair = self.transform(image=image, mask=mask)

        return pair['image'], pair['mask'].unsqueeze(0), int(name.replace('.png', ''))

    def __len__(self):
        return len(self.samples)


# def get_image_num(image_root):
def give_augmentations(config, train):
    if train == True:
        augmentations = A.Compose([
            A.Normalize(),
            A.Resize(config.image_size, config.image_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.1),
            # A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])
    else:
        augmentations = A.Compose([
            A.Normalize(),
            A.Resize(config.image_size, config.image_size, interpolation=cv2.INTER_NEAREST),
            ToTensorV2()
        ])
    return augmentations

    

def get_kfold_multimodal_dataloader(config, fold=3):

    seed = 42

    # 创建保存划分结果的文件夹
    split_dir = "split"
    os.makedirs(os.path.join(split_dir, config.finetune.model_choose), exist_ok=True)

    # config = config.dataset.Breast_US
    data_root = config.dataset.Breast_US.data_root
    swt_root = data_root + 'Time-Color'
    swv_root = data_root + 'Velocity-Color'
    bus_root = data_root  + 'Velocity-Gray'
    gt_root = data_root  + 'GroundTruth'

    all_samples = [name for name in os.listdir(swt_root) if name[0] != "."]

    kf = KFold(n_splits=fold, shuffle=True, random_state=seed)

    train_loaders, val_loaders = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_samples)):

        train_samples = np.array(all_samples)[train_idx]
        val_samples = np.array(all_samples)[val_idx]

        # 分别保存训练样本和验证样本到不同的CSV文件
        # 训练样本
        train_df = pd.DataFrame({
            'sample': train_samples,
            'fold': [fold] * len(train_samples),
            'split': ['train'] * len(train_samples)
        })
        train_df.to_csv(os.path.join(split_dir, config.finetune.model_choose, f'fold_{fold}_train.csv'), index=False)

        # 验证样本
        val_df = pd.DataFrame({
            'sample': val_samples,
            'fold': [fold] * len(val_samples),
            'split': ['val'] * len(val_samples)
        })
        val_df.to_csv(os.path.join(split_dir, config.finetune.model_choose, f'fold_{fold}_val.csv'), index=False)

        train_augmentation = give_augmentations(config.dataset.Breast_US, train=True)
        val_augmentation = give_augmentations(config.dataset.Breast_US, train=False)

        train_dataset_swt = Dataset(swt_root, gt_root, train_augmentation, train_samples)
        val_dataset_swt = Dataset(swt_root, gt_root, val_augmentation, val_samples)

        train_dataset_swv = Dataset(swv_root, gt_root, train_augmentation, train_samples)
        val_dataset_swv = Dataset(swv_root, gt_root, val_augmentation, val_samples)

        train_dataset_bus = Dataset(bus_root, gt_root, train_augmentation, train_samples)
        val_dataset_bus = Dataset(bus_root, gt_root, val_augmentation, val_samples)


        train_loader = data.DataLoader(list(zip(train_dataset_swt, train_dataset_swv, train_dataset_bus)),
                                       batch_size=config.dataset.Breast_US.batch_size,
                                       shuffle=True,
                                       num_workers=config.dataset.Breast_US.num_workers,
                                       pin_memory=True)
        val_loader = data.DataLoader(list(zip(val_dataset_swt, val_dataset_swv, val_dataset_bus)),
                                       batch_size=config.dataset.Breast_US.batch_size,
                                       shuffle=False,
                                       num_workers=config.dataset.Breast_US.num_workers,
                                       pin_memory=True)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders


def get_multi_center_dataloader(config):
    config = config.dataset.Breast_US
    train_root = config.train_root
    test_root = config.test_root

    train_time_color_root = train_root + 'Time-Color'
    test_time_color_root = test_root + 'Time-Color'

    train_velocity_color_root = train_root + 'Velocity-Color'
    test_velocity_color_root = test_root + 'Velocity-Color'

    train_velocity_gray_root = train_root + 'Velocity-Gray'
    test_velocity_gray_root = test_root + 'Velocity-Gray'

    train_gt_root = train_root + 'GroundTruth'
    test_gt_root = test_root + 'GroundTruth'

    train_augmentation = give_augmentations(config, train=True)
    test_augmentation = give_augmentations(config, train=False)

    train_dataset_time_color = Multi_Center_Dataset(train_time_color_root, train_gt_root, train_augmentation)
    test_dataset_time_color = Multi_Center_Dataset(test_time_color_root, test_gt_root, test_augmentation)

    train_dataset_velocity_color = Multi_Center_Dataset(train_velocity_color_root, train_gt_root, train_augmentation)
    test_dataset_velocity_color = Multi_Center_Dataset(test_velocity_color_root, test_gt_root, test_augmentation)

    train_dataset_velocity_gray = Multi_Center_Dataset(train_velocity_gray_root, train_gt_root, train_augmentation)
    test_dataset_velocity_gray = Multi_Center_Dataset(test_velocity_gray_root, test_gt_root, test_augmentation)

    train_loader = data.DataLoader(
        list(zip(train_dataset_time_color, train_dataset_velocity_color, train_dataset_velocity_gray)),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True)
    val_loader = data.DataLoader(
        list(zip(test_dataset_time_color, test_dataset_velocity_color, test_dataset_velocity_gray)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True)

    return train_loader, val_loader

if __name__ == '__main__':


    config = EasyDict(yaml.load(open('../config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    config.data_root = '../datasets/cs/seg/Breast-US'
    train_loader, test_loader = get_kfold_multimodal_dataloader(config)
    train_num = 0
    for i, image_batch in enumerate(train_loader):
        modal_TC, modal_VC, modal_VG = image_batch
        if i == 0:
            print(modal_TC[0])
        print(modal_TC[0].size())
        print(modal_TC[1].size())
        train_num += 1
    test_num = 0
    for i, image_batch in enumerate(test_loader):
        modal_TC, modal_VC, modal_VG = image_batch
        if i == 0:
            print(modal_TC[0])
        print(modal_TC[0].size())
        print(modal_TC[1].size())
        test_num += 1
    print(train_num)
    print(test_num)
    print(train_num + test_num)


    # train_loader, val_loader = get_unimodal_dataloader(config)



