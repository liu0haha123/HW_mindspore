import mindspore.nn as nn
import mindspore


class VGGStyleDiscriminator128(nn.Cell):
    """VGG style discriminator with input size 128 x 128.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch, num_feat):
        super(VGGStyleDiscriminator128, self).__init__()

        self.conv0_0 = nn.Conv2d(
            num_in_ch, num_feat, 3, 1, padding=1, has_bias=True, pad_mode="pad"
        )
        self.conv0_1 = nn.Conv2d(
            num_feat, num_feat, 4, 2, padding=1, has_bias=False, pad_mode="pad"
        )
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(
            num_feat, num_feat * 2, 3, 1, padding=1, has_bias=False, pad_mode="pad"
        )
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            num_feat * 2, num_feat * 2, 4, 2, padding=1, has_bias=False, pad_mode="pad"
        )
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(
            num_feat * 2, num_feat * 4, 3, 1, padding=1, has_bias=False, pad_mode="pad"
        )
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 4, 2, padding=1, has_bias=False, pad_mode="pad"
        )
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(
            num_feat * 4, num_feat * 8, 3, 1, padding=1, has_bias=False, pad_mode="pad"
        )
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, padding=1, has_bias=False, pad_mode="pad"
        )
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.linear1 = nn.Dense(num_feat * 8 * 4 * 4 * 4, 100)
        self.linear2 = nn.Dense(100, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()

    def construct(self, x):

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(
            self.bn0_1(self.conv0_1(feat))
        )  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(
            self.bn1_1(self.conv1_1(feat))
        )  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(
            self.bn2_1(self.conv2_1(feat))
        )  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (8, 8)


        feat = self.flatten(feat)
        feat = self.lrelu(self.linear1(feat))

        out = self.linear2(feat)
        return out
