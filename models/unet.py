import mindspore.nn as nn
import mindspore.ops as ops


class DoubleConv(nn.Cell):
    """两个3x3卷积结构"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.SequentialCell(
            # 原文使用了valid, 这里使用对same构成对称结构
            nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def construct(self, x):
        logits = self.double_conv(x)
        return logits


class Down(nn.Cell):
    """最大池+双卷积"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def construct(self, x):
        logits = self.maxpool_conv(x)
        return logits


class Up(nn.Cell):
    """反卷积+concat+双卷积"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Conv2dTranspose(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        # _CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # 裁剪
        # x2 = ops.pad(x2, )
        x = ops.concat([x2, x1], axis=1)
        logits = self.conv(x)
        return logits


class UNet(nn.Cell):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # 下采样
        self.input = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        # 上采样
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        # 输出
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def construct(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits
