import torch
import torch.nn as nn

# Pyramid Pooling Module
class PyramidPoolingModule(nn.Module):

    def __init__(self, size, in_channels):
        super(PyramidPoolingModule, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels//4, 1, stride=1, padding=0, bias=True),
            nn.Upsample(size=size, mode='bilinear')
        )

        self.layer_2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(2),
            nn.Conv2d(in_channels, in_channels//4, 1, stride=1, padding=0, bias=True),
            nn.Upsample(size=size, mode='bilinear')
        )

        self.layer_3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(3),
            nn.Conv2d(in_channels, in_channels//4, 1, stride=1, padding=0, bias=True),
            nn.Upsample(size=size, mode='bilinear')
        )

        self.layer_4 = nn.Sequential(
            nn.AdaptiveMaxPool2d(6),
            nn.Conv2d(in_channels, in_channels//4, 1, stride=1, padding=0, bias=True),
            nn.Upsample(size=size, mode='bilinear')
        )

    def forward(self, feature):

        feature_1 = self.layer_1(feature)
        feature_2 = self.layer_2(feature)
        feature_3 = self.layer_3(feature)
        feature_4 = self.layer_4(feature)

        output = torch.cat([feature, feature_4, feature_3, feature_2, feature_1], dim=1)

        return output

# Bottleneck
class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, in_channels, inner_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels*4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels*4)
        )

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.layer(x) + residual
        out = self.relu(out)

        return out

# Pyramid Scene Parsing Network
class PSPNet_ResNet50(nn.Module):

    def __init__(self, in_size=256, num_class=2):
        super(PSPNet_ResNet50, self).__init__()

        self.num_class = num_class

        # ResNet-50
        self.in_channels = 64

        self.resnet_proc = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.resnet_layer1 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(Bottleneck, 64, 3, stride=1)
        )
        self.resnet_layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.resnet_layer3 = self._make_layer(Bottleneck, 256, 6, stride=1)
        self.resnet_layer4 = self._make_layer(Bottleneck, 512, 3, stride=1)

        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(size=(in_size//8, in_size//8), in_channels=2048)

        # Output Layer
        self.output_layer = nn.Conv2d(2048*2, self.num_class, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_channels, n_blocks, stride=1):

        downsample = None
        if stride != 1 or self.in_channels != n_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, n_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, n_channels, stride, downsample))
        self.in_channels = n_channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, n_channels))

        return nn.Sequential(*layers)

    def forward(self, images):

        feature = self.resnet_proc( images )
        feature = self.resnet_layer1( feature )
        feature = self.resnet_layer2( feature )
        feature = self.resnet_layer3( feature )
        feature = self.resnet_layer4( feature )

        feature = self.ppm( feature )

        labels = self.output_layer( feature )

        return labels

if __name__=='__main__':

    # 1. Build model
    net = PSPNet_ResNet50()
    net = nn.DataParallel(net, device_ids=[0])
    print(net)

    # 2. Fake image batch
    images = torch.rand((16, 1, 256, 256))

    # 3. Network forward
    labels_pred = net(images)

    # 4. Info
    print(labels_pred.shape)