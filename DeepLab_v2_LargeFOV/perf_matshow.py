import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from dataset.data import ICH_CT_32
from model.DeepLab_v2_LargeFOV import DeepLab_LargeFOV_VGG16D
from tqdm import tqdm

import matplotlib.pyplot as plt

def compute_matshow(net, dataloader):

    # Setting
    net.eval()

    # Record
    n_cls = dataloader.dataset.CLASS_NUM
    # N = [ [ TN, FP ], [ FN, TP ] ]
    N = np.zeros((n_cls, n_cls))
    with torch.no_grad():
        for i, (images_batch, labels_batch) in enumerate(tqdm(dataloader)):

            labels_batch_pred = net(images_batch).cpu()
            labels_batch_pred = F.interpolate(labels_batch_pred, size=labels_batch.shape[2:4], mode='bilinear', align_corners=False)
            
            labels_batch_prob = F.softmax(labels_batch_pred, dim=1)[:,1,:,:].contiguous().view(-1).numpy()
            labels_batch_np = labels_batch.view(-1).numpy()

            thresh = 0.5
            for idx in range(labels_batch_np.shape[0]):
                if labels_batch_prob[idx] > thresh:
                    N[int(labels_batch_np[idx]), 1] += 1
                else:
                    N[int(labels_batch_np[idx]), 0] += 1

    N[1,0] *= 85.0
    N[1,1] *= 85.0
    # Plot
    print(N)
    plt.matshow(N)
    plt.show()

if __name__=='__main__':

    # 1. Load dataset
    dataset_eval = ICH_CT_32(
        ROOT='../data/Processed',
        transform=T.Compose( [T.ToTensor(), T.Normalize([0.5,], [0.5,])] ),
        is_train=False,
        fold_idx=1)
    dataloader_eval = DataLoader(dataset_eval, batch_size=16, shuffle=True, num_workers=1)

    # 2. Build model
    net = DeepLab_LargeFOV_VGG16D()
    net = nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load('weights/DeepLab_v2_FOV.00999.pkl'))

    # 3. ROC and AUC
    compute_matshow(net, dataloader_eval)
