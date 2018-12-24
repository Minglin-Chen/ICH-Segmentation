import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial Path
class SpatialPath(nn.Module):

    def __init__(self):
        super(SpatialPath, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=2, padding=1, bias=True), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):

        return self.layer(images)

# Attention Refinment Module
class ARM(nn.Module):

    def __init__(self, in_channels):
        super(ARM, self).__init__()

        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, feature):

        return (feature * self.layer(feature))

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

# Context Path
class ContextPath_ResNet50(nn.Module):

    def __init__(self):
        super(ContextPath_ResNet50, self).__init__()

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
        self.resnet_layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.resnet_layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Global Average Pooling
        self.GAP = nn.AdaptiveAvgPool2d(1)

        # Attention Refinment Module
        self.ARM_16 = ARM(in_channels=1024)
        self.ARM_32 = ARM(in_channels=2048)

        # Deploy U-shape structure to fuse the features
        self.layer_conv_32 = nn.Conv2d(2048*2, 1024, 3, stride=1, padding=1, bias=True)
        self.layer_conv_16 = nn.Conv2d(1024*2, 512, 3, stride=1, padding=1, bias=True)

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

        feature_2 = self.resnet_proc( images )
        feature_4 = self.resnet_layer1( feature_2 )
        feature_8 = self.resnet_layer2( feature_4 )
        feature_16 = self.resnet_layer3( feature_8 )
        feature_32 = self.resnet_layer4( feature_16 )
        feature_bin = self.GAP( feature_32 )

        output = F.interpolate(feature_bin, size=images.shape[3]//32, mode='bilinear')

        feature_32 = self.ARM_32( feature_32 )
        output = self.layer_conv_32( torch.cat([output, feature_32], dim=1) )
        output = F.interpolate(output, scale_factor=2, mode='bilinear')

        feature_16 = self.ARM_16( feature_16 )
        output = self.layer_conv_16( torch.cat([output, feature_16], dim=1) )
        output = F.interpolate(output, scale_factor=2, mode='bilinear')

        return output

# Feature Fusion Module
class FFM(nn.Module):

    def __init__(self, in_channels):
        super(FFM, self).__init__()

        self.proc = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.layer_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, spatial_feature, context_feature):

        feature_cat = torch.cat([spatial_feature, context_feature], dim=1)
        feature_fused = self.proc(feature_cat)

        return ( feature_fused * self.layer_attention(feature_fused) + feature_fused )

# BiSeNet
class BiSeNet_ResNet50(nn.Module):

    def __init__(self, num_class=2):
        super(BiSeNet_ResNet50, self).__init__()

        # Spatial Path
        self.spatial_path = SpatialPath()
        # Context Path
        self.context_path = ContextPath_ResNet50()
        # Feature Fusion Mudule
        self.ffm = FFM(in_channels=512*2)

        # Output
        self.layer_output = nn.Conv2d(256, num_class, 1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images):

        spatial_feature = self.spatial_path(images)
        context_feature = self.context_path(images)
        fused_feature = self.ffm(spatial_feature, context_feature)
        labels = self.layer_output(fused_feature)

        return labels

if __name__=='__main__':

    # 1. Build model
    net = BiSeNet_ResNet50()
    net = nn.DataParallel(net, device_ids=[0])
    print(net)

    # 2. Fake image batch
    images = torch.rand((16, 1, 256, 256))

    # 3. Network forward
    labels_pred = net(images)

    # 4. Info
    print(labels_pred.shape)