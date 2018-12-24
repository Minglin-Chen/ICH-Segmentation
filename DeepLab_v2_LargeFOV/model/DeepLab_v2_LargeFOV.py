import torch
import torch.nn as nn

from collections import OrderedDict

class DeepLab_LargeFOV_VGG16D(nn.Module):

    def __init__(self, num_class=2, init_weights=True):
        super(DeepLab_LargeFOV_VGG16D, self).__init__()
        self.num_class = num_class

        self.basenet_vgg16d = nn.Sequential(
            # Layer-1 (224 x 224 x 3 -> 112 x 112 x 64)
            nn.Conv2d(1, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            # Layer-2 (112 x 112 x 64 -> 56 x 56 x 128)
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            # Layer-3 (56 x 56 x 128 -> 28 x 28 x 256)
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            # Layer-4 (28 x 28 x 256 -> (28+1) x (28+1) x 512)
            nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1, padding=1),
            # Layer-5 (29 x 29 x 512 -> (29+1) x (29+1) x 512)
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1, padding=1)
        )

        self.largeFOV_layer = nn.Sequential(
            # Layer-6 (30 x 30 x 512 -> 30 x 30 x 1024) -- Atrous Convolution
            nn.Conv2d(512, 1024, 3, stride=1, padding=12, dilation=12), nn.ReLU(inplace=True),
            # Layer-7 (30 x 30 x 1024 -> 30 x 30 x 1024)
            nn.Conv2d(1024, 1024, 1, stride=1, padding=0), nn.ReLU(inplace=True),
            # Layer-8 (30 x 30 x 1024 -> 30 x 30 x self.num_class)
            nn.Conv2d(1024, self.num_class, 1, stride=1, padding=0)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, image):
        feature = self.basenet_vgg16d(image)
        label = self.largeFOV_layer(feature)
        return label
    
    def decode_pred(self, label_pred, dst_size):
        label_pred = F.upsample(label_pred, size=dst_size, mode='bilinear', align_corners=False)
        label_pred = torch.argmax(label_pred, dim=1)
        return label_pred

    def finetune_from(self, PATH):
        pretrained_dict = torch.load(PATH)
        refine_dict = OrderedDict({
            'basenet_vgg16d.0.weight' : pretrained_dict['features.0.weight'],
            'basenet_vgg16d.0.bias' : pretrained_dict['features.0.bias'],
            'basenet_vgg16d.2.weight' : pretrained_dict['features.2.weight'],
            'basenet_vgg16d.2.bias' : pretrained_dict['features.2.bias'],
            'basenet_vgg16d.5.weight' : pretrained_dict['features.5.weight'],
            'basenet_vgg16d.5.bias' : pretrained_dict['features.5.bias'],
            'basenet_vgg16d.7.weight' : pretrained_dict['features.7.weight'],
            'basenet_vgg16d.7.bias' : pretrained_dict['features.7.bias'],
            'basenet_vgg16d.10.weight' : pretrained_dict['features.10.weight'],
            'basenet_vgg16d.10.bias' : pretrained_dict['features.10.bias'],
            'basenet_vgg16d.12.weight' : pretrained_dict['features.12.weight'],
            'basenet_vgg16d.12.bias' : pretrained_dict['features.12.bias'],
            'basenet_vgg16d.14.weight' : pretrained_dict['features.14.weight'],
            'basenet_vgg16d.14.bias' : pretrained_dict['features.14.bias'],
            'basenet_vgg16d.17.weight' : pretrained_dict['features.17.weight'],
            'basenet_vgg16d.17.bias' : pretrained_dict['features.17.bias'],
            'basenet_vgg16d.19.weight' : pretrained_dict['features.19.weight'],
            'basenet_vgg16d.19.bias' : pretrained_dict['features.19.bias'],
            'basenet_vgg16d.21.weight' : pretrained_dict['features.21.weight'],
            'basenet_vgg16d.21.bias' : pretrained_dict['features.21.bias'],
            'basenet_vgg16d.24.weight' : pretrained_dict['features.24.weight'],
            'basenet_vgg16d.24.bias' : pretrained_dict['features.24.bias'],
            'basenet_vgg16d.26.weight' : pretrained_dict['features.26.weight'],
            'basenet_vgg16d.26.bias' : pretrained_dict['features.26.bias'],
            'basenet_vgg16d.28.weight' : pretrained_dict['features.28.weight'],
            'basenet_vgg16d.28.bias' : pretrained_dict['features.28.bias']
        })

        self.load_state_dict(refine_dict, strict=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__=='__main__':

    # 1. Build the network
    # deeplab = DeepLab_LargeFOV_VGG16D(num_class=151, init_weights=True)
    net = DeepLab_LargeFOV_VGG16D(num_class=2, init_weights=True)
    # deeplab.finetune_from('weights/vgg16-397923af.pth')

    # 2. Fake image batch
    image = torch.rand((8, 3, 256, 256))

    # 3. Network forward
    label_pred = deeplab(image)

    # 4. Info
    print(label_pred.shape)