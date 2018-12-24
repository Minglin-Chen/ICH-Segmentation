import torch
import torch.nn as nn
import torch.nn.functional as F

def unet_block(in_channels, out_channels):

    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1), nn.ReLU(inplace=True)
    )
    return block

class UNet(nn.Module):

    def __init__(self, num_class=2, block=unet_block, init_weights=True):
        super(UNet, self).__init__()
        self.num_class = num_class

        # Build
        self.left_layer_1 = block(1, 64)
        self.left_layer_2 = nn.Sequential( nn.MaxPool2d(2), block(64, 128) )
        self.left_layer_3 = nn.Sequential( nn.MaxPool2d(2), block(128, 256) )
        self.left_layer_4 = nn.Sequential( nn.MaxPool2d(2), block(256, 512) )

        self.bottom_layer = nn.Sequential( nn.MaxPool2d(2), block(512, 1024), nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) )

        self.right_layer_4 = nn.Sequential( block(1024, 512), nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) )
        self.right_layer_3 = nn.Sequential( block(512, 256), nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) )
        self.right_layer_2 = nn.Sequential( block(256, 128), nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) )
        self.right_layer_1 = block(128, 64)

        self.output_layer = nn.Conv2d(64, self.num_class, 1, stride=1, padding=0)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image):

        left_feature_1 = self.left_layer_1( image )
        left_feature_2 = self.left_layer_2( left_feature_1 )
        left_feature_3 = self.left_layer_3( left_feature_2 )
        left_feature_4 = self.left_layer_4( left_feature_3 )
        bottom_feature = self.bottom_layer( left_feature_4 )
        right_feature_4 = self.right_layer_4( torch.cat([ left_feature_4, bottom_feature ], dim=1) )
        right_feature_3 = self.right_layer_3( torch.cat([ left_feature_3, right_feature_4 ], dim=1) )
        right_feature_2 = self.right_layer_2( torch.cat([ left_feature_2, right_feature_3 ], dim=1) )
        right_feature_1 = self.right_layer_1( torch.cat([ left_feature_1, right_feature_2 ], dim=1) )
        output = self.output_layer( right_feature_1 )

        return output

if __name__=='__main__':

    #net = UNet()

    #torch.save(net, 'unet.pth')
    net_load = torch.load('unet.pth')

    print(type(net_load))