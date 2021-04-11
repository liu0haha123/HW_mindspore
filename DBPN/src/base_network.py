import math
import mindspore
import mindspore.nn as nn
class DenseBlock(nn.Cell):
    def __init__(self,input_size,output_size,bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.norm = norm
        self.fc = nn.Dense(input_size,output_size,has_bias=bias)
        if self.norm =='batch':
            self.bn = nn.BatchNorm1d(output_size)
        else:
            # print("minspore不支持InstanceNorm")
            self.bn = nn.BatchNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, pad_mode='pad', padding=padding, has_bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2d(output_size)
        else:
            # print("minspore不支持InstanceNorm")
            self.bn = nn.BatchNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Conv2dTranspose(input_size, output_size, kernel_size, stride, pad_mode='pad', padding=padding, has_bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        else:
           # print("minspore不支持InstanceNorm")
           self.bn = nn.BatchNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, pad_mode='pad', padding=padding, has_bias=bias)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, pad_mode='pad', padding=padding, has_bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        else:
            # print("minspore不支持InstanceNorm")
            self.bn = nn.BatchNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)
        add_ops = mindspore.ops.Add()
        out = add_ops(out, residual)
        return out


class UpBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='relu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def construct(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0-x)
        return h1 + h0


class Upsampler(nn.Cell):
    # prelu不支持GPU
    def __init__(self, scale, n_feat, bn=False, act='relu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            PixelShuffle = mindspore.ops.DepthToSpace(2)
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias=bias, activation=None, norm=None))
            modules.append(PixelShuffle())
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            # modules.append(torch.nn.PReLU())
        self.up = nn.SequentialCell(*modules)

        self.activation = act
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class UpBlockPix(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='relu', norm=None):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)

    def construct(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='relu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def construct(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlockPix(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='relu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)

    def construct(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='relu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def construct(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class DownBlockPix(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,bias=True, activation='relu', norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def construct(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='relu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def construct(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlockPix(nn.Cell):
    # prelu不支持GPU
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='relu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def construct(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class PSBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding=padding, has_bias=bias)
        self.ps = mindspore.ops.DepthToSpace(2)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        else:
            # print("minspore不支持InstanceNorm")
            self.bn = nn.BatchNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def construct(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(nn.Cell):
    # prelu不支持GPU
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            # 需要计算大小
            Upsample = mindspore.ops.ResizeBilinear((input_size//scale_factor, input_size//scale_factor))
            self.upsample = nn.SequentialCell(
                Upsample(),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def construct(self, x):
        out = self.upsample(x)
        return out

