#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import torchvision
import torch.utils.data as torch_data
from torchvision import transforms
from utils import BaseEnumOptions


class ImageInterpolation(BaseEnumOptions):
    """
    继承自BaseEnumOptions，定义图像插值方法的枚举选项。
    """
    nearest = transforms.InterpolationMode.NEAREST      # 最近邻插值
    box = transforms.InterpolationMode.BOX              # box插值
    bilinear = transforms.InterpolationMode.BILINEAR    # 双线性插值
    hamming = transforms.InterpolationMode.HAMMING      # hamming插值
    bicubic = transforms.InterpolationMode.BICUBIC      # 双三次插值
    lanczos = transforms.InterpolationMode.LANCZOS      # lanczos插值


class ImageNetDataLoaders(object):
    """
    数据加载器的提供者，专门用于加载ImageNet数据集中的训练集和验证集。
    该类负责定义数据预处理的转换，创建训练和验证的数据加载器。并提供懒加载机制，确保数据加载器只在需要时才被初始化。
    Data loader provider for ImageNet images, providing a train and a validation loader.
    It assumes that the structure of the images is
        images_dir
            - train
                - label1
                - label2
                - ...
            - val
                - label1
                - label2
                - ...
    """

    def __init__(
        self,
        images_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        interpolation: transforms.InterpolationMode,
    ):
        """
        Parameters
        ----------
        images_dir: str
            Root image directory
        image_size: int
            Number of pixels the image will be re-sized to (square)
        batch_size: int
            Batch size of both the training and validation loaders
        num_workers
            Number of parallel workers loading the images
        interpolation: transforms.InterpolationMode
            Desired interpolation to use for resizing.
        """

        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # For normalization, mean and std dev values are calculated per channel
        # and can be found on the web.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # 定义数据标准化的转换

        self.train_transforms = transforms.Compose(                 # 定义训练集的数据预处理转换
            [
                transforms.RandomResizedCrop(image_size, interpolation=interpolation.value),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.val_transforms = transforms.Compose(                   # 定义验证集的数据预处理转换
            [
                transforms.Resize(image_size + 24, interpolation=interpolation.value),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # 用于懒加载训练和验证的数据加载器
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self) -> torch_data.DataLoader:
        """
        用于懒加载训练集的数据加载器，只有在第一次访问时才会初始化数据加载器。
        """
        if not self._train_loader:                                  # 如果训练集的数据加载器还未初始化
            root = os.path.join(self.images_dir, "train")           # 训练集的根目录
            train_set = torchvision.datasets.ImageFolder(root, transform=self.train_transforms)     # 创建训练集数据集
            self._train_loader = torch_data.DataLoader(             # 创建训练集的数据加载器
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._train_loader

    @property
    def val_loader(self) -> torch_data.DataLoader:
        """
        用于懒加载验证集的数据加载器，只有在第一次访问时才会初始化数据加载器。
        """
        if not self._val_loader:                                    # 如果验证集的数据加载器还未初始化
            root = os.path.join(self.images_dir, "val")             # 验证集的根目录
            val_set = torchvision.datasets.ImageFolder(root, transform=self.val_transforms)         # 创建验证集数据集
            self._val_loader = torch_data.DataLoader(               # 创建验证集的数据加载器
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._val_loader
