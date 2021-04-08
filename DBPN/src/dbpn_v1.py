import math
import mindspore
import mindspore.nn as nn
import numpy as np
from base_network import *

class Net(nn.Cell):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        # prelu不支持GPU
        # self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        # self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='relu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='relu', norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        self.down7 = D_DownBlock(base_filter, kernel, stride, padding, 7)
        self.up8 = D_UpBlock(base_filter, kernel, stride, padding, 7)
        self.down8 = D_DownBlock(base_filter, kernel, stride, padding, 8)
        self.up9 = D_UpBlock(base_filter, kernel, stride, padding, 8)
        self.down9 = D_DownBlock(base_filter, kernel, stride, padding, 9)
        self.up10 = D_UpBlock(base_filter, kernel, stride, padding, 9)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        # for m in self.modules():
        for m in self.cells_and_names():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                mindspore.common.initializer.HeNormal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                mindspore.common.initializer.HeNormal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def construct(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        op = mindspore.ops.Concat(1)

        concat_h = op((h2, h1))
        l = self.down2(concat_h)

        concat_l = op((l, l1))
        h = self.up3(concat_l)

        concat_h = op((h, concat_h))
        l = self.down3(concat_h)

        concat_l = op((l, concat_l))
        h = self.up4(concat_l)

        concat_h = op((h, concat_h))
        l = self.down4(concat_h)

        concat_l = op((l, concat_l))
        h = self.up5(concat_l)

        concat_h = op((h, concat_h))
        l = self.down5(concat_h)

        concat_l = op((l, concat_l))
        h = self.up6(concat_l)

        concat_h = op((h, concat_h))
        l = self.down6(concat_h)

        concat_l = op((l, concat_l))
        h = self.up7(concat_l)

        concat_h = op((h, concat_h))
        l = self.down7(concat_h)

        concat_l = op((l, concat_l))
        h = self.up8(concat_l)

        concat_h = op((h, concat_h))
        l = self.down8(concat_h)

        concat_l = op((l, concat_l))
        h = self.up9(concat_l)

        concat_h = op((h, concat_h))
        l = self.down9(concat_h)

        concat_l = op((l, concat_l))
        h = self.up10(concat_l)

        concat_h = op((h, concat_h))
        x = self.output_conv(concat_h)

        return x
