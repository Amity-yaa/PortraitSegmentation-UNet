"""
2022-06-14 11:08
    经过实验，感觉这个收敛很看初始化，很难优化，因此决定修改加入Residual结构。
    另外，之后有空时应该实验一下，将pytorch模型转为onnx模型，检查一下网络拓扑图是否能够识别到中间列表内的结构，怀疑这些不会被识别到，甚至不能进行反向转播。
    另外仔细研究了一下，torchvision.resnet都是直接将多个block写成self.layer1,self.layer2,...,self.layer5的形式。
"""
import torch
from torch import nn
from torch.nn import functional as F


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True
    if classname.find('BatchNorm2d') != -1:
        print(classname)
        m.inplace = True


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=64, activation=nn.ReLU, sigmoid_output=True, residual=False):
        super().__init__()
        self.down_blocks = []
        self.up_blocks = []
        self.upsamplings = []
        self.activation = activation
        out_channels_first = out_channels
        self.down_block1 = UNetBlock(in_channels, out_channels, activation=activation, residual=residual)
        self.up_block1 = UNetBlock(out_channels * 2, out_channels, activation=activation, residual=residual)
        self.upsampling1 = UpsamplingBlock(out_channels * 2, out_channels)
        in_channels = out_channels
        out_channels *= 2

        self.down_block2 = UNetBlock(in_channels, out_channels, activation=activation, residual=residual)
        self.up_block2 = UNetBlock(out_channels * 2, out_channels, activation=activation, residual=residual)
        self.upsampling2 = UpsamplingBlock(out_channels * 2, out_channels)
        in_channels = out_channels
        out_channels *= 2

        self.down_block3 = UNetBlock(in_channels, out_channels, activation=activation, residual=residual)
        self.up_block3 = UNetBlock(out_channels * 2, out_channels, activation=activation, residual=residual)
        self.upsampling3 = UpsamplingBlock(out_channels * 2, out_channels)
        in_channels = out_channels
        out_channels *= 2

        self.down_block4 = UNetBlock(in_channels, out_channels, activation=activation, residual=residual)
        self.up_block4 = UNetBlock(out_channels * 2, out_channels, activation=activation, residual=residual)
        self.upsampling4 = UpsamplingBlock(out_channels * 2, out_channels)
        in_channels = out_channels
        out_channels *= 2

        # for i in range(4):
        #     self.down_blocks.append(
        #         # self.create_block(in_channels, out_channels)
        #         UNetBlock(in_channels, out_channels).to(device)
        #         # 对于会存储到列表中的对象，必须手动放置到gpu上。
        #         # 而直接属于UNet类的属性不需要手动放到gpu上。后来所有都加了to(device)是为了防止以后外面忘了加或以为外面不用指定device
        #     )
        #     self.up_blocks.append(
        #         nn.Sequential(
        #             UNetBlock(out_channels * 2, out_channels).to(device)
        #             # self.create_block(out_channels * 2, out_channels)
        #         )
        #     )
        #     self.upsamplings.append(UpsamplingBlock(out_channels * 2, out_channels).to(device))
        #     in_channels = out_channels
        #     out_channels = out_channels * 2
        # self.bottleneck = self.create_block(512, 1024)
        self.bottleneck = UNetBlock(in_channels, out_channels, activation=activation, residual=residual)
        self.max_pooling = nn.MaxPool2d(2, 2)
        if sigmoid_output:
            self.conv1x1 = nn.Conv2d(out_channels_first, 1, 1)  # ?原论文甚至不过softmax？！Emmm，但是模型结构是末尾叫softmaxLoss，所以其实是有的。
        else:
            self.conv1x1 = nn.Conv2d(out_channels_first, 2, 1)
        self.softmax = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_output = sigmoid_output

    # def tensorboard_add(self, tensorboard):
    #
    # def create_block(self, in_channels, out_channels):
    #     conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    #     conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    #     block = nn.Sequential(conv1, conv2)
    #     return block

    # def create_upsampling_block(self, in_channels, out_channels):
    #     conv1 = nn.Conv2d(in_channels, out_channels, 1)
    #     us = nn.Upsample(scale_factor=2)
    #     block = nn.Sequential(conv1, us)
    #     return block

    def forward(self, x):
        # outputs = []
        # for i in range(len(self.down_blocks)):  # https://blog.csdn.net/m0_48095841/article/details/120909598
        #     x = self.down_blocks[i](x)
        #     outputs.append(x)
        #     x = self.max_pooling(x)
        x1 = self.down_block1(x)
        x = self.max_pooling(x1)
        x2 = self.down_block2(x)
        x = self.max_pooling(x2)
        x3 = self.down_block3(x)
        x = self.max_pooling(x3)
        x4 = self.down_block4(x)
        x = self.max_pooling(x4)
        x = self.bottleneck(x)
        x = self.upsampling4(x, x4)
        x = self.up_block4(x)
        x = self.upsampling3(x, x3)
        x = self.up_block3(x)
        x = self.upsampling2(x, x2)
        x = self.up_block2(x)
        x = self.upsampling1(x, x1)
        x = self.up_block1(x)
        # for i in range(len(self.up_blocks)):
        #     x1 = outputs[-1 - i]
        #     # print(i, x.shape, x1.shape)
        #     x = self.upsamplings[-1 - i](x, x1)
        #     x = self.up_blocks[-1 - i](x)
        x = self.conv1x1(x)
        if self.sigmoid_output:
            x = self.sigmoid(x)
        else:
            x = self.softmax(x)
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, residual=False):
        super(UNetBlock, self).__init__()
        conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        bn = nn.BatchNorm2d(out_channels)
        act = activation()
        self.block = nn.Sequential(conv1, bn, act, conv2, bn, act)  # block的决定代码

        if residual and (in_channels != out_channels):
            self.residual_convbn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        self.residual = residual

    def forward(self, x):
        x1 = x
        x = self.block(x)
        if hasattr(self, 'residual_convbn'):
            x1 = self.residual_convbn(x1)
        if self.residual:
            x = x1 + x
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        # self.us = nn.Upsample(scale_factor=2)
        # 1080经过3次下采样后变成了135，不是偶数了，于是第4次下采样后就变成了67，67如果直接放大两倍会变成134，导致无法矩阵拼接，
        # 所以后面采用动态的interpolate，这样保证了任意形状的输入

    def forward(self, x, x1):
        x = self.conv1(x)
        x = F.interpolate(x, x1.shape[2:])
        # print(x.shape, x1.shape)
        x = torch.cat([x, x1], dim=1)
        return x


if __name__ == '__main__':
    # data = torch.ones(1, 3, 1920, 1080).cuda()
    # data = torch.ones(1, 3, 1280, 720).cuda()
    data = torch.ones(1, 3, 640, 360).cuda()
    model = UNet(3)  # .cuda()
    # model = model.apply(inplace_relu)
    output = model(data)
    print('input shape:', data.shape)
    print('output shape:', output.shape)
