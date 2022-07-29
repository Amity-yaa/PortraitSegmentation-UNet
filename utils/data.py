import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
import numpy as np
from torchvision import transforms
import random
from .transforms import *


### https://ifwind.github.io/2021/11/03/Pytorch与深度学习自查手册2-数据加载和预处理/#dataset类

# class MadaDataset(Dataset):
#     def __init__(self, root=r'C:\Users\meiyan\Desktop\VirtualBackground\dataset\train'):
#         def get_paths(folder):
#             folder_root = os.path.join(root, folder)
#             mp4s = os.listdir(folder_root)
#             mp4s_root = [os.path.join(folder_root, i) for i in mp4s]
#             paths_ = []
#             for mp4_root in mp4s_root:
#                 imgs = os.listdir(mp4_root)
#                 imgs = [i for i in imgs if '.jpg' in i]  # 只取jpg格式图片即可，到时mask直接改文件名为png格式图片即可。
#                 paths_ += imgs
#             return paths_
#
#         indoor_paths = get_paths('indoor')
#         outdoor_paths = get_paths('outdoor')
#         self.paths = indoor_paths + outdoor_paths
#
#
#     def __len__(self):
#         return len(self.paths)
#
#     def __getitem__(self, idx):
#         img_path = self.paths[idx]
#         mask_path = img_path.replace('.jpg','.png')
#

# 为了保证mask和image同时做数据增强，还是要自己手写数据增强
# Data_Transform = transforms.Compose([
#
# ])



class VirtualBG_Dataset(Dataset):
    def __init__(self,
                 root=r'C:\Users\meiyan\Desktop\VirtualBackground\dataset\resized\train',
                 transform=None):  # , uniform_size=False):
        super().__init__()
        self.paths = glob.glob(root + '/image/*.png')
        # self.uniform_size = uniform_size
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        mask_path = img_path.replace('image', 'alpha')

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img.shape[-1] != 3:
            print(img_path, img.shape)
        # mask = mask / 255 # 这一步交给torchvision.transforms.ToTensor
        if isinstance(self.transform, Transforms):
            img, mask = self.transform([img, mask])
            # print('After Transforming', img.shape, mask.shape)
        else:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask

        # if self.uniform_size:
        #     H, W = self.uniform_size
        #     img_h, img_w, _ = img.shape
        #     if img_h/img_w > H/W:  # 说明高度多了，要补宽
        #         new_w = img_h*W/H
        #         w_add = new_w - img_w


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from transforms import *
    transform = Transforms()
    dataset = VirtualBG_Dataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=2)
    for x, y in dataloader:
        print(x.shape, y.shape)
        img = x[0, 0].numpy()
        label = y[0, 0].numpy()
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(label)
        plt.show()
