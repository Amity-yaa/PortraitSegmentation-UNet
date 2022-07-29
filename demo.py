from torchsummary import summary
from torch import nn
import numpy as np
import cv2
import torch
from models.Unet import UNet
import os

if __name__ == '__main__':
    model_root = 'log/models'
    # pretrained_model = 'UNet_sigmoid_20220615-180225/Model_UNet_sigmoid_Epoch-4_Iters-6875_loss-0.2109_criterionLoss-0.2315_IoULoss-0.3106_IoU-1.1694.pth'
    # pretrained_model = 'UNet_sigmoid_20220620-164053/Model_UNet_sigmoid_Epoch-19_Iters-13800_loss-0.3284_criterionLoss-0.3780_IoULoss-0.6739_IoU-0.3947.pth'
    # pretrained_model = r'UNet_sigmoid_20220625-211838/Model_UNet_sigmoid_Epoch-39_Iters-16080_loss-0.3619_criterionLoss-0.3364_IoULoss-0.8521_IoU-0.1600.pth'
    # pretrained_model = 'UNet_sigmoid_20220625-211838/Model_UNet_sigmoid_Epoch-39_Iters-16080_loss-0.3619_criterionLoss-0.3364_IoULoss-0.8521_IoU-0.1600.pth'
    pretrained_model = 'UNet_sigmoid_20220625-211838/Model_UNet_sigmoid_Epoch-39_Iters-16080_loss-0.3619_criterionLoss-0.3364_IoULoss-0.8521_IoU-0.1600.pth'
    # pretrained_model = 'UNet_sigmoid_20220626-173340/Model_UNet_sigmoid_Epoch-30_Iters-12462_loss-0.0022_criterionLoss-0.0170_IoULoss-0.1384_IoU-0.8707.pth'
    # pretrained_model = 'UNet_sigmoid_20220627-011210/Model_UNet_sigmoid_Epoch-78_Iters-31758_loss-0.0011_criterionLoss-0.0152_IoULoss-0.1118_IoU-0.8942.pth'
    sigmoid_output = False  # Softmax的显存占用要高于Sigmoid
    load_optim = True
    device = 'cpu'

    model = UNet(3, out_channels=8, sigmoid_output=sigmoid_output, residual=True, norm=nn.InstanceNorm2d).to(device)
    if pretrained_model:
        pretrained_model = os.path.join(model_root, pretrained_model)
        model.load_state_dict(torch.load(pretrained_model), strict=False)

    summary(model, (3, 320, 180), device=device)
    bg = cv2.imread('bg.png')[..., :3]
    bg = cv2.resize(bg, (320, 180))
    cap = cv2.VideoCapture(0)
    # model.eval()
    with torch.no_grad():
        while True:
            ret, img = cap.read()
            img_bk = img.copy()
            img = cv2.resize(img, (320, 180))
            img = img.transpose((2, 0, 1))[None, :, :, :]
            img = torch.tensor(img / 255, dtype=torch.float32).to(device)
            result = model(img)
            if sigmoid_output:
                result = result.cpu().numpy()[0, 0]
            else:
                result = result.cpu().numpy()[0, 1]
            mask = result.copy()
            mask = np.array([mask] * 3).transpose((1, 2, 0))
            img_merge = (1 - mask) * bg + mask * cv2.resize(img_bk, (320, 180))
            img_merge = img_merge.astype(np.uint8)
            result *= 255
            result = result.astype(np.uint8)
            cv2.imshow("img", img_bk)
            cv2.imshow("result", result)
            cv2.imshow("merge", img_merge)
            cv2.waitKey(1)
