import mindspore.nn as nn
import mindspore.ops as ops
from src.resnet import resnet50, resnet101


class PPM(nn.Cell):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            # PyramidPool
            self.features.append(
                nn.SequentialCell(
                    # nn.AdaptiveAvgPool2d(bin)
                    #nn.AvgPool2d(bin) 这里暂时没有功能对应的实现
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, has_bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(),
                )
            )

        self.features = nn.CellList(self.features)

    def construct(self, x):
        x_size = x.shape()
        out = [x]
        """
        for f in self.features:
            out.append(
                F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True)
            )
        """
        for f in self.features:
            resize_interpolate = ops.ResizeBilinear(x_size[2:], align_corners=True)
            out.append(resize_interpolate(f(x)))
        cat = ops.Concat(axis=1)
        return cat(out)


class PSPNet(nn.Cell):
    def __init__(
        self,
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=2,
        zoom_factor=8,
        use_ppm=True,
        criterion=nn.SoftmaxCrossEntropyWithLogits(sparse=True),
        pretrained=True,
    ):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        if layers == 50:
            resnet = resnet50()
        elif layers == 101:
            resnet = resnet101()
        else:
            print("mindspore 不支持其他规格的ResNet")

        self.layer1 = nn.SequentialCell(resnet.conv1, resnet.bn1, resnet.maxpool)
        self.layer2, self.layer3, self.layer4, self.layer5 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        for n, m in self.layer3.cells_and_names():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "down_sample_layer.0" in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif "down_sample_layer.0" in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.SequentialCell(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(512, classes, kernel_size=1),
        )
        if self.training:
            self.aux = nn.SequentialCell(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(256, classes, kernel_size=1),
            )

    def construct(self, x, y=None):
        x_size = x.shape()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_tmp = self.layer4(x)
        x = self.layer5(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            resize_interpolate = ops.ResizeBilinear(size=(h, w), align_corners=True)
            x = resize_interpolate.interpolate(x)
        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                resize_interpolate = ops.ResizeBilinear(size=(h, w), align_corners=True)
                aux = resize_interpolate.interpolate(aux)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x