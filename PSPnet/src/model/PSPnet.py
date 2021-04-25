import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from src.model.resnet import resnet50
from mindspore.train.serialization import load_param_into_net,load_checkpoint
import mindspore.common.initializer as weight_init
from mindspore import dtype as mstype
# 加注释去掉的部分都是ModelArts不支持的，本地train可以用

class ResNet(nn.Cell):
    def __init__(self, pretrained_path, pretrained=False):
        super(ResNet, self).__init__()
        resnet = resnet50(1001)
        if pretrained:
            param_dict = load_checkpoint(pretrained_path)
            load_param_into_net(resnet, param_dict)
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

    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_aux = self.layer4(x)
        x = self.layer5(x_aux)

        return x_aux, x


class _PSPModule(nn.Cell):
    def __init__(self, in_channels, pool_sizes, norm_layer, feature_shape):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        # -----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        # -----------------------------------------------------#
        self.stages = nn.CellList(
            [
                self._make_stages(in_channels, out_channels, norm_layer, pool_size)
                for pool_size in pool_sizes
            ]
        )
        self.cat = ops.Concat(axis=1)
        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.SequentialCell(
            nn.Conv2d(
                in_channels + (out_channels * len(pool_sizes)),
                out_channels,
                kernel_size=3,
                padding=1,
                has_bias=False,
                pad_mode="pad",
            ),
            norm_layer(out_channels),
            nn.ReLU(),
        )
        # 这里要预先指定输入的大小
        self.feature_shape = feature_shape
        self.resize_ops = ops.ResizeBilinear(
            (self.feature_shape[0], self.feature_shape[1]), True
        )

    def _make_stages(self, in_channels, out_channels, norm_layer, bin_sz):
        prior = nn.AvgPool2d(kernel_size=bin_sz, stride=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU()

        return nn.SequentialCell(prior, conv, bn, relu)

    def construct(self, features):
        pyramids = [features]
        for stage in self.stages:
            pyramids.append(self.resize_ops(stage(features)))
        pyramids_tuple = (
            pyramids[0],
            pyramids[1],
            pyramids[2],
            pyramids[3],
            pyramids[4],
        )
        output = self.cat(pyramids_tuple)
        output = self.bottleneck(output)
        return output


class PSPNet(nn.Cell):
    def __init__(
        self,
        pool_sizes=[1, 2, 3, 6],
        feature_size=15,
        num_classes=21,
        backbone="resnet50",
        pretrained=True,
        pretrained_path="",
        aux_branch=True,
    ):
        """
        """
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone == "resnet50":
            self.backbone = ResNet(
                pretrained=pretrained, pretrained_path=pretrained_path
            )
            aux_channel = 1024
            out_channel = 2048
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use resnet50 .".format(backbone)
            )
        self.feature_shape = [feature_size, feature_size]
        self.pool_sizes = [feature_size - pool_size for pool_size in pool_sizes]
        # --------------------------------------------------------------#
        # 	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        # --------------------------------------------------------------#
        self.master_branch = nn.SequentialCell(
            _PSPModule(
                out_channel,
                self.pool_sizes,
                norm_layer=norm_layer,
                feature_shape=self.feature_shape,
            ),
            nn.Conv2d(out_channel // 4, num_classes, kernel_size=1),
        )

        self.aux_branch = aux_branch
        # self.dropout = ops.Dropout2D(0.1)
        if self.aux_branch:
            # ---------------------------------------------------#
            # 	利用特征获得预测结果
            # ---------------------------------------------------#
            self.auxiliary_branch = nn.SequentialCell(
                nn.Conv2d(
                    aux_channel,
                    out_channel // 8,
                    kernel_size=3,
                    padding=1,
                    has_bias=False,
                    pad_mode="pad",
                ),
                norm_layer(out_channel // 8),
                nn.ReLU(),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1),
            )
        self.resize = nn.ResizeBilinear()
        self.shape = ops.Shape()
        #self.init_weights(self.master_branch)

    def init_weights(self, *models):
        for model in models:
            for _, cell in model.cells_and_names():
                if isinstance(cell, nn.Conv2d):
                    cell.weight.set_data(
                        weight_init.initializer(
                            weight_init.HeNormal(), cell.weight.shape, cell.weight.dtype
                        )
                    )
                if isinstance(cell, nn.Dense):
                    cell.weight.set_data(
                        weight_init.initializer(
                            weight_init.TruncatedNormal(0.01),
                            cell.weight.shape,
                            cell.weight.dtype,
                        )
                    )
                    cell.bias.set_data(1e-4, cell.bias.shape, cell.bias.dtype)

    def construct(self, x):
        x_shape = self.shape(x)
        x_aux, x = self.backbone(x)
        output = self.master_branch(x)
        output = self.resize(output, size=(x_shape[2:4]))
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            #output_aux,mask = self.dropout(output_aux)
            output_aux = self.resize(output_aux, size=(x_shape[2:4]))
            return output_aux, output
        else:
            return output
