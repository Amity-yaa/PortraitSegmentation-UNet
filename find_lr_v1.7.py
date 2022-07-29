"""
version 1.5
    在1.4的基础上，添加如下功能： 当指定了pretrained_model时，将会找到对应的log给tensorboard续写。(因此完善了Iters的接续)
"""

import os.path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # https://blog.csdn.net/weixin_55593481/article/details/123617710
import torch.optim.optimizer
from torch import nn
from utils.data import VirtualBG_Dataset
from utils.transforms import Transforms
from torch.utils.data import DataLoader
from models.Unet import UNet
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from utils.utilities import Logger
import shutil
from utils.find_lr import LRFinder

if __name__ == '__main__':
    pretrained_model = None  # r'C:\Users\meiyan\Desktop\VirtualBackground\log\models\UNet_sigmoid_20220612-121150\Model_UNet_sigmoid_Epoch-20_Iters-64806_loss-0.9470.pth'
    sigmoid_output = False  # Softmax的显存占用要高于Sigmoid
    transform = Transforms(target_shape=(320, 180), convert2gray=0)
    train_dataloader = DataLoader(VirtualBG_Dataset(transform=transform),
                                  batch_size=14, shuffle=True)
    model = UNet(3, out_channels=8, sigmoid_output=sigmoid_output,
                 activation=nn.ReLU, residual=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-3, weight_decay=0.005)

    if sigmoid_output:
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    lr_finder = LRFinder(model, train_dataloader, optimizer, criterion, 1e-6, 10, 'cuda')
    lr_finder.find_lr()
    lr_finder.plot()
