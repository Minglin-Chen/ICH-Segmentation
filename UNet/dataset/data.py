import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid

if __name__=='__main__':
    from augmentation import random_flip_pair, ramdom_rotation_pair
else:
    from .augmentation import random_flip_pair, ramdom_rotation_pair

class ICH_CT_32(Dataset):

    def __init__(self, ROOT, IMAGE_SIZE=None, LABEL_SIZE=None, transform=None, is_train=True, fold_idx=1):
        self.ROOT = ROOT
        self.IMAGE_SIZE = IMAGE_SIZE
        self.LABEL_SIZE = LABEL_SIZE
        self.transform = transform
        self.is_train = is_train

        self.items = []
        with open(os.path.join(self.ROOT, 'train_tumor_fold_{}.txt'.format(fold_idx) if self.is_train else 'val_tumor_fold_{}.txt'.format(fold_idx)), 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            image_path, label_path = line.rstrip('\n').split(' ')
            self.items.append([image_path, label_path])

        self.CLASS_NUM = 2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        [image_path, label_path] = self.items[index]

        # 1. Load image & label
        image = cv2.imread(os.path.join(self.ROOT, image_path), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(self.ROOT, label_path), cv2.IMREAD_GRAYSCALE)

        # 2. Data Augmentation
        if self.is_train:
            image, label = random_flip_pair(image, label)
            image, label = ramdom_rotation_pair(image, label)
		
        if self.IMAGE_SIZE is not None:
            image = cv2.resize(image, self.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        image = image[:,:,np.newaxis]
        if self.transform is not None:
            image = self.transform(image)
        
        if self.LABEL_SIZE is not None:
            label = cv2.resize(label, self.LABEL_SIZE, interpolation=cv2.INTER_NEAREST)
        label[label!=0]=1
        label = torch.tensor(label, dtype=torch.long)

        return image, label

if __name__=='__main__':

    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)

    # 1. Load dataset
    dataset = ICH_CT_32(
        ROOT='../../data/Processed',
        transform=T.Compose( [T.ToTensor(), T.Normalize((0.5,), (0.5,))] ),
        is_train=True,
        fold_idx=1)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    # 2. Loop
    for i, (images_batch, labels_batch) in enumerate(dataloader):

        images_tensor = make_grid(images_batch, nrow=4, padding=12, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=1.0)
        images_np = images_tensor.numpy().transpose((1,2,0))

        labels_tensor = make_grid(labels_batch.unsqueeze(dim=1), nrow=4, padding=12, normalize=False, pad_value=1.0)
        labels_np = labels_tensor.numpy().transpose((1,2,0)).astype(np.float)

        cv2.imshow('image', images_np)
        cv2.imshow('label', labels_np)
        cv2.waitKey()

        break