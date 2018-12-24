import torch
import torch.nn as nn

# Residual Conv Unit
class RCU(nn.Module):

    def __init__(self, in_channels, conv_channels):
        super(RCU, self).__init__()

        self.layer = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, conv_channels, 3, stride=1, padding=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, in_channels, 3, stride=1, padding=1, bias=False)
        )

    def forward(self, feature):

        return feature + self.layer(feature)

# Multi-resolution Fusion
class MRF(nn.Module):

    def __init__(self, hr_in_channels, lr_in_channels):
        super(MRF, self).__init__()

        self.hr_layer = nn.Conv2d(hr_in_channels, hr_in_channels, 3, stride=1, padding=1, bias=False)

        self.lr_layer = nn.Sequential(
            nn.Conv2d(lr_in_channels, hr_in_channels, 3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
    
    def forward(self, hr_feature, lr_feature):
        return self.hr_layer(hr_feature) + self.lr_layer(lr_feature)

# Chained Residual Pooling
class CRP(nn.Module):

    def __init__(self, in_channels, N=2):
        super(CRP, self).__init__()

        self.proc = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList(
            [ nn.Sequential(
                nn.MaxPool2d(5, stride=1, padding=2), 
                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False)) for i in range(N)
                ]
        )

    def forward(self, feature):
        
        feature_ret = self.proc(feature)
        feature_in = feature_ret

        for layer in self.layers:
            feature_in = layer(feature_in)
            feature_ret = feature_ret + feature_in
        
        return feature_ret
        
# RefineNet Block
class RefineNet_Block(nn.Module):

    def __init__(self, hr_in_channels, lr_in_channels=None, idx=4):
        super(RefineNet_Block, self).__init__()
        self.idx = idx

        # Two/One Input Path 2xRCU (Residual Conv Unit)
        self.hr_2RCU = nn.Sequential(
            RCU(in_channels=hr_in_channels, conv_channels=512 if self.idx==4 else 256),
            RCU(in_channels=hr_in_channels, conv_channels=512 if self.idx==4 else 256)
        )
        
        if self.idx != 4:
            self.lr_2RCU = nn.Sequential(
                RCU(in_channels=lr_in_channels, conv_channels=256),
                RCU(in_channels=lr_in_channels, conv_channels=256)
            )
        
        # Multi-resolution Fusion
        if self.idx != 4:
            self.layer_MRF = MRF(hr_in_channels, lr_in_channels)

        # Chained Residual Pooling
        self.layer_CRP = CRP(hr_in_channels, N=2)
        
        # Output Conv
        if self.idx == 1:
            self.conv_out = nn.Sequential(
                RCU(in_channels=hr_in_channels, conv_channels=256),
                RCU(in_channels=hr_in_channels, conv_channels=256),
                RCU(in_channels=hr_in_channels, conv_channels=256)
            )
        else:
            self.conv_out = RCU(in_channels=hr_in_channels, conv_channels=256)

    def forward(self, hr_feature, lr_feature=None):

        if self.idx != 4:
            hr_feature = self.hr_2RCU(hr_feature)
            lr_feature = self.lr_2RCU(lr_feature)
            feature = self.layer_MRF(hr_feature, lr_feature)
        else:
            feature = self.hr_2RCU(hr_feature)

        feature = self.layer_CRP(feature)
        feature = self.conv_out(feature)

        return feature

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

# RefineNet
class RefineNet_ResNet152(nn.Module):

    def __init__(self, num_class=2):
        super(RefineNet_ResNet152, self).__init__()

        self.num_class = num_class

        # ResNet-152
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
        self.resnet_layer2 = self._make_layer(Bottleneck, 128, 8, stride=2)
        self.resnet_layer3 = self._make_layer(Bottleneck, 256, 36, stride=2)
        self.resnet_layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # RefineNet Block
        self.RefineNet_Block4 = RefineNet_Block(hr_in_channels=2048, lr_in_channels=None, idx=4)
        self.RefineNet_Block3 = RefineNet_Block(hr_in_channels=1024, lr_in_channels=2048, idx=3)
        self.RefineNet_Block2 = RefineNet_Block(hr_in_channels=512, lr_in_channels=1024, idx=2)
        self.RefineNet_Block1 = RefineNet_Block(hr_in_channels=256, lr_in_channels=512, idx=1)

        # Output Layer
        self.output_layer = nn.Conv2d(256, self.num_class, 1, stride=1, padding=0)

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

        feature_2 = self.resnet_proc( images )
        feature_4 = self.resnet_layer1( feature_2 )
        feature_8 = self.resnet_layer2( feature_4 )
        feature_16 = self.resnet_layer3( feature_8 )
        feature_32 = self.resnet_layer4( feature_16 )
        
        inner_feature = self.RefineNet_Block4( feature_32 )
        inner_feature = self.RefineNet_Block3( feature_16, inner_feature )
        inner_feature = self.RefineNet_Block2( feature_8, inner_feature )
        inner_feature = self.RefineNet_Block1( feature_4, inner_feature )

        labels = self.output_layer(inner_feature)

        return labels

if __name__=='__main__':

    # 1. Build model
    net = RefineNet_ResNet152()
    net = nn.DataParallel(net, device_ids=[0])
    print(net)

    # 2. Fake image batch
    images = torch.rand((16, 1, 256, 256))

    # 3. Network forward
    labels_pred = net(images)

    # 4. Info
    print(labels_pred.shape)