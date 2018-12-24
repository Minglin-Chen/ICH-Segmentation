import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as T

from tensorboard_logger import Logger

from dataset.data import ICH_CT_32
from model.BiSeNet import BiSeNet_ResNet50
from utils.ops import train_op, eval_op

import json

config = {
    'dataset_root': '../data/Processed',
    'batch_size': 16,
    'lr': 1e-3,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_epoch': 2,
    'log_folder': 'logs/',
    'save_folder': 'weights/'}

if not os.path.exists( config['log_folder'] ):
    os.mkdir( config['log_folder'] )
if not os.path.exists( config['save_folder'] ):
    os.mkdir( config['save_folder'] )

def train(fold_idx=1):

    # 1. Load dataset
    dataset_train = ICH_CT_32(
        ROOT=config['dataset_root'],
        transform=T.Compose( [T.ToTensor(), T.Normalize([0.5,], [0.5,])] ),
        is_train=True,
        fold_idx=fold_idx)
    dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, num_workers=1)

    dataset_eval = ICH_CT_32(
        ROOT=config['dataset_root'],
        transform=T.Compose( [T.ToTensor(), T.Normalize([0.5,], [0.5,])] ),
        is_train=False,
        fold_idx=fold_idx)
    dataloader_eval = DataLoader(dataset_eval, batch_size=config['batch_size'], shuffle=False, num_workers=1)

    # 2. Build model
    net = BiSeNet_ResNet50()
    # net.finetune_from('pretrained_weights/vgg16-397923af.pth')
    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
    print(net)

    # 3. Criterion
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 85.0]))

    # 4. Optimizer
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    # 5. Tensorboard logger
    logger_train = Logger(logdir=os.path.join(config['log_folder'], 'fold_{}'.format(fold_idx), 'train'), flush_secs=2)
    logger_eval = Logger(logdir=os.path.join(config['log_folder'], 'fold_{}'.format(fold_idx), 'eval'), flush_secs=2)

    # 6. Train loop
    DSC_MAX, IOU1_MAX, sensitivity_MAX, specificity_MAX = -1.0, -1.0, -1.0, -1.0
    for epoch in range(config['num_epoch']):

        train_op(net, dataloader_train, criterion, optimizer, scheduler, epoch, logger_train)
        DSC, IOU1, sensitivity, specificity = eval_op(net, dataloader_eval, criterion, epoch, logger_eval)

        torch.save(net.state_dict(), os.path.join(config['save_folder'], 'RefineNet.newest.{}.pkl'.format(fold_idx)))
        
        if DSC_MAX <= DSC:
            DSC_MAX = DSC
            torch.save(net.state_dict(), os.path.join(config['save_folder'], 'RefineNet.{}.pkl'.format(fold_idx)))
        if IOU1_MAX <= IOU1: IOU1_MAX = IOU1
        if sensitivity_MAX <= sensitivity: sensitivity_MAX = sensitivity
        if specificity_MAX <= specificity: specificity_MAX = specificity

    return DSC_MAX, IOU1_MAX, sensitivity_MAX, specificity_MAX, DSC, IOU1, sensitivity, specificity


if __name__=='__main__':

    # Cross Validation
    DSC_MAX_list, IOU1_MAX_list, sensitivity_MAX_list, specificity_MAX_list = [], [], [], []
    DSC_list, IOU1_list, sensitivity_list, specificity_list = [], [], [], []
    for i in range(5):
        DSC_MAX, IOU1_MAX, sensitivity_MAX, specificity_MAX, DSC, IOU1, sensitivity, specificity = train(fold_idx=i+1)

        DSC_MAX_list.append(DSC_MAX)
        IOU1_MAX_list.append(IOU1_MAX)
        sensitivity_MAX_list.append(sensitivity_MAX)
        specificity_MAX_list.append(specificity_MAX)

        DSC_list.append(DSC)
        IOU1_list.append(IOU1)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
    
    # Output
    content = {
        'DSC_MAX': DSC_MAX_list,
        'IOU1_MAX': IOU1_MAX_list,
        'sensitivity_MAX': sensitivity_MAX_list,
        'specificity_MAX': specificity_MAX_list,
        'DSC': DSC_list,
        'IOU1': IOU1_list,
        'sensitivity': sensitivity_list,
        'specificity': specificity_list }
    with open('cv_perf_result.json', 'w') as f:
        json.dump(content, f)