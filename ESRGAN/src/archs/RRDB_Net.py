import mindspore
import mindspore.nn as nn
import mindspore.ops.functional as F
import functools


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.SequentialCell(*layers)


class ResidualDenseBlock_5C(nn.Cell):
    def __init__(self, nf=64, gc=32,res_beta=0.2, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, padding=1, has_bias=bias,pad_mode="pad")
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, padding=1, has_bias=bias,pad_mode="pad")
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, padding=1, has_bias=bias,pad_mode="pad")
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, padding=1, has_bias=bias,pad_mode="pad")
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, padding=1, has_bias=bias,pad_mode="pad")
        self.lrelu = nn.LeakyReLU(0.2)
        self.res_beta = res_beta
        self.cat = mindspore.ops.Concat(1)

    def construct(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(self.cat((x, x1))))
        x3 = self.lrelu(self.conv3(self.cat((x, x1, x2))))
        x4 = self.lrelu(self.conv4(self.cat((x, x1, x2, x3))))
        x5 = self.conv5(self.cat((x, x1, x2, x3, x4)))
        return x5 * self.res_beta + x


class RRDB(nn.Cell):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32,res_beta=0.2):
        super(RRDB, self).__init__()
        self.res_beta = res_beta
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Cell):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, padding=1, has_bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, padding=1, has_bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, padding=1, has_bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, padding=1, has_bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, padding=1, has_bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, padding=1, has_bias=True)

        self.lrelu = nn.LeakyReLU(0.2)

    def construct(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        #fea_size = mindspore.ops.Shape(fea)
        fea_size = fea.Shape()
        resize_interpolate_1 = mindspore.ops.ResizeBilinear(size=(2 * fea_size[-2], 2 * fea_size[-1]))
        resize_interpolate_2 = mindspore.ops.ResizeBilinear(size=(4 * fea_size[-2], 4 * fea_size[-1]))
        fea = self.lrelu(self.upconv1(resize_interpolate_1(fea)))
        fea = self.lrelu(self.upconv2(resize_interpolate_2(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
