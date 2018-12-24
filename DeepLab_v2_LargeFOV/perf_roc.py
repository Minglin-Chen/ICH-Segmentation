import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from dataset.data import ICH_CT_32
from model.DeepLab_v2_LargeFOV import DeepLab_LargeFOV_VGG16D
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def compute_ROC_and_AUC(net, dataloader):

    # Setting
    net.eval()

    # Record
    target = []
    prob = []
    with torch.no_grad():

        for i, (images_batch, labels_batch) in enumerate(tqdm(dataloader)):

            labels_batch_pred = net(images_batch).cpu()
            labels_batch_pred = F.interpolate(labels_batch_pred, size=labels_batch.shape[2:4], mode='bilinear', align_corners=False)
            labels_batch_prob = F.softmax(labels_batch_pred, dim=1)[:,1,:,:]

            target += labels_batch.view(-1).tolist()
            prob += labels_batch_prob.contiguous().view(-1).tolist()
    
    # Compute ROC curve and AUC value
    fpr, tpr, threshold = roc_curve(target, prob, pos_label=1)
    auc_value = auc(fpr, tpr)

    # Plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='DeepLab v2 LargeFOV (AUC = {:.2f})'.format(auc_value))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ( Receiver Operating Characteristic )')
    plt.legend(loc="lower right")
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
    net.load_state_dict(torch.load('weights/DeepLab_v2_FOV.00000.pkl'))

    # 3. ROC and AUC
    compute_ROC_and_AUC(net, dataloader_eval)
