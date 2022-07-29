"""
本脚本用于在命令行通过参数指定输入文件和指定图像，然后通过plt呈现预测结果，目前更主要是用于绘制实验中model.eval()的影响。
同时编写此脚本时，顺道整理了单张图像的输入需要经过的处理，整理进了utils.utilities.py中，该处理时BGR2Tensor。
"""

import argparse
import numpy as np
import torch
from torch import nn
import os
import cv2
from models.Unet import UNet
from utils.utilities import BGR2Tensor
from matplotlib import pyplot as plt

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # args.add_argument('--model_root', default=r'log\models\UNet_sigmoid_20220620-164053', type=str)
    args.add_argument('--model_root', default=r'log\models\UNet_sigmoid_20220625-211838', type=str)
    # args.add_argument('--model_file',
    #                   default=r'Model_UNet_sigmoid_Epoch-39_Iters-27600_loss-0.3556_criterionLoss-0.3766_IoULoss-0.6784_IoU-0.3880.pth',
    #                   type=str)
    args.add_argument('--model_file',
                      default=r'Model_UNet_sigmoid_Epoch-39_Iters-16080_loss-0.3619_criterionLoss-0.3364_IoULoss-0.8521_IoU-0.1600.pth',
                      type=str)
    args.add_argument('--input', default=r'qianxi.jpg', type=str)
    args.add_argument('--device', default='cuda', type=str)
    args = args.parse_args()

    origin = cv2.imread(args.input, cv2.IMREAD_COLOR)
    origin = cv2.resize(origin, (320, 180))
    img = BGR2Tensor(origin, args.device)

    model_path = os.path.join(args.model_root, args.model_file)
    sigmoid_output = False
    model = UNet(3, out_channels=8, sigmoid_output=sigmoid_output,
                 activation=nn.ReLU, residual=True, norm=nn.InstanceNorm2d).to(args.device)
    model_params = torch.load(model_path)
    model.load_state_dict(model_params)

    model.train()
    result_train = model(img)[0, 1].detach().cpu().numpy()

    model.eval()
    result_eval = model(img)[0, 1].detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(origin, cv2.COLOR_BGR2RGB))
    plt.title('origin')
    plt.subplot(132)
    plt.imshow(result_train)
    plt.title('mode Train')
    plt.subplot(133)
    plt.imshow(result_eval)
    plt.title('mode Eval')
    plt.savefig('output_' + args.input)
    plt.show()
