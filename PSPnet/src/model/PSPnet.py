import mindspore.nn as nn
import mindspore.ops as ops
from src.model.resnet import resnet50
import mindspore.ops.functional as F
from mindspore import load_checkpoint,load_param_into_net


class ResNet(nn.Cell):
    def __init__(self,pretrained_path,pretrained=False):
        super(ResNet, self).__init__()
        resnet = resnet50(1000)
        if pretrained:
            # 这个需要到现场拿ImageNet训练 预训练的是CIFAR-10 数据集太小
            param_dict = load_checkpoint(pretrained_path)
            param_not_load = load_param_into_net(resnet, param_dict)

        self.layer1 = nn.SequentialCell(resnet.conv1, resnet.bn1, resnet.maxpool)
        self.layer2, self.layer3, self.layer4, self.layer5 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        for n, m in self.layer4.cells_and_names():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "down_sample_layer.0" in n:
                m.stride = (1, 1)
        for n, m in self.layer5.cells_and_names():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif "down_sample_layer.0" in n:
                m.stride = (1, 1)

    def construct(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_aux = self.layer4(x)
        x = self.layer5(x_aux)

        return x_aux,x

class _PSPModule(nn.Cell):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        # -----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        # -----------------------------------------------------#
        self.stages = nn.CellList(
            [
                self._make_stages(in_channels, out_channels, pool_size, norm_layer)
                for pool_size in pool_sizes
            ]
        )
        self.cat  = ops.Concat(axis=1)
        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.SequentialCell(
            nn.Conv2d(
                in_channels + (out_channels * len(pool_sizes)),
                out_channels,
                kernel_size=3,
                padding=1,
                has_bias=False,
                pad_mode="pad"
            ),
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            #nn.Dropout2d(0.1),
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AvgPool2d(kernel_size=1)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU()
        return nn.SequentialCell(prior, conv, bn, relu)

    def construct(self, features):
        feature_size= list(features.shape)
        h, w = feature_size[2], feature_size[3]
        pyramids = [features]
        pyramids.extend(
            [   F.partial()
                #F.interpolate(stage(features), size=(h, w), mode="bilinear", align_corners=True)
                for stage in self.stages
            ]
        )
        output = self.bottleneck(self.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Cell):
    def __init__(
        self,
        num_classes=21,
        downsample_factor=8,
        backbone="resnet50",
        pretrained=True,
        aux_branch=True,
    ):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone == "resnet50":
            self.backbone = ResNet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use resnet50.".format(backbone)
            )

        # --------------------------------------------------------------#
        # 	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        # --------------------------------------------------------------#
        self.master_branch = nn.SequentialCell(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel // 4, num_classes, kernel_size=1),
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            # ---------------------------------------------------#
            # 	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            # ---------------------------------------------------#
            self.auxiliary_branch = nn.SequentialCell(
                nn.Conv2d(
                    aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False
                ),
                norm_layer(out_channel // 8),
                nn.ReLU(),
                #nn.Dropout2d(0.1),
                nn.Dropout(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1),
            )

    def forward(self, x):
        x_size = list(x.shape)
        input_size = (x_size[2], x_size[3])
        resize_ops = ops.ResizeBilinear(size=input_size,align_corners=True)
        x_aux, x = self.backbone(x)
        output = self.master_branch(x)
        output = resize_ops(output)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = resize_ops(
                output_aux
            )
            return output_aux, output
        else:
            return output
