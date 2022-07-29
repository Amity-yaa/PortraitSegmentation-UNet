import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np


class LRFinder:
    def __init__(self, model, dataloader, optimizer, criterion, lr_begin, lr_end, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_begin = lr_begin
        self.lr_end = lr_end
        self.device = device
        self.loss = []
        self.lr = []
        pass

    def find_lr(self):
        batch_num = len(self.dataloader)
        multi = (self.lr_end / self.lr_begin) ** (1 / batch_num)
        self.loss = []
        self.lr = []

        lr = self.lr_begin
        for i, (x, y) in enumerate(self.dataloader):
            lr *= multi
            self.optimizer.param_groups[0]['lr'] = lr

            if isinstance(self.criterion, nn.CrossEntropyLoss):
                y = 1 * (y >= 0.5)
                y = y.long()  # https://www.csdn.net/tags/OtDaIg5sMTU1NzktYmxvZwO0O0OO0O0O.html 不需要自己转为one-hot编码，不过这样子得自己降低一个维度
                # N_y, C_y, H_y, W_y = y.shape
                y = torch.squeeze(y, dim=1)
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model(x)
            loss = self.criterion(out, y)

            if self.model.sigmoid_output:
                inner = out * y
                union = out + y - inner
                sum_axis = 2
            else:
                inner = torch.squeeze(out[:, 1::2], 1) * y
                union = torch.squeeze(out[:, 1::2], 1) + y - inner
                sum_axis = 1
            iou = inner.sum(axis=sum_axis).sum(axis=sum_axis) / union.sum(axis=sum_axis).sum(axis=sum_axis)
            iou = iou.mean()
            iou_loss = -torch.log(iou)  # https://blog.csdn.net/c2250645962/article/details/106053242

            loss = loss + iou_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss.append(loss.detach().cpu().item())
            self.lr.append(lr)
            print('{}/{}, loss:{:.4f}, lr:{}'.format(i, len(self.dataloader), loss.detach().cpu().item(), lr))



    def plot(self):
        lr_log = np.log10(self.lr)
        plt.plot(lr_log, self.loss)
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.show()
