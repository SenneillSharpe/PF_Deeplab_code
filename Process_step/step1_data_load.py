# 用于处理图像和对应的分割标签
import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


# ThreeTrainDataset用于训练集
class ThreeTrainDataset(Dataset):
    def __init__(self, image_dir, segmentation_label_dir):
        """
        初始化函数
        :param image_dir:
        :param segmentation_label_dir:
        """
        self.image_dir = image_dir
        self.label_dir = segmentation_label_dir
        self.files_name = os.listdir(self.image_dir)
        # self.image_dir 和 self.label_dir 分别存储图像目录和标签目录的路径。
        # self.files_name 存储图像目录中的所有文件名列表。

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.files_name[index])
        label_path = os.path.join(self.label_dir, self.files_name[index])

        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (256, 256))
        image = np.transpose(image, (2, 0, 1))
        image = np.float32(image)

        label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
        label = cv.resize(label, (256, 256))
        label = label.reshape(1, 256, 256)
        label = np.float32(label)

        # 转换为Tensor，并进行归一化
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.float32) / 255.0

        return image, label


# ThreeNonTrainDataset主要用于验证（validation）和测试（test）数据集，其中 state 参数用于区分是验证集还是测试集
class ThreeNonTrainDataset(Dataset):
    def __init__(self, image_dir, segmentation_label_dir, state):
        """
        初始化函数
        :param image_dir:
        :param segmentation_label_dir:
        :param state:
        """
        self.image_dir = image_dir
        self.label_dir = segmentation_label_dir
        self.files_name = os.listdir(self.image_dir)
        self.state = state

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, index):
        image_name = self.files_name[index]

        image_path = os.path.join(self.image_dir, self.files_name[index])
        label_path = os.path.join(self.label_dir, self.files_name[index])

        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (256, 256))
        image = np.transpose(image, (2, 0, 1))
        image = np.float32(image)

        label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
        label = cv.resize(label, (256, 256))
        label = label.reshape(1, 256, 256)
        label = np.float32(label)

        # 转换为Tensor，并进行归一化
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.float32) / 255.0

        if self.state == "val":
            return image, label
        elif self.state == "test":
            return image, image_name


if __name__ == '__main__':
    three_channel_train_dataset = ThreeTrainDataset(
        image_dir="../Data2/train_image",
        segmentation_label_dir="../Data2/train_label",
    )

    three_channel_non_train_dataset = ThreeNonTrainDataset(
        image_dir="../Data2/val_image",
        segmentation_label_dir="../Data2/val_label",
        state="val"
    )
