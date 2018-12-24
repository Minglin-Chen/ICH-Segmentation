import time
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm

def train_op(net, dataloader, criterion, optimizer, scheduler, epoch, logger):

    # Setting
    net.train()

    # Loop
    for i, (images_batch, labels_batch) in enumerate(dataloader):
        
        start_time = time.clock()
        # 1. Calculate loss
        labels_batch_pred = net(images_batch)
        labels_batch_pred = F.interpolate(labels_batch_pred, size=labels_batch.shape[2:4], mode='bilinear', align_corners=False)
        loss = criterion(labels_batch_pred.cpu(), labels_batch)

        # 2. Update the weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
            
        # 3. Logger
        logger.log_value('loss', loss.item(), step=i+epoch*len(dataloader))
        print('Epoch {:0>5} [{}/{}]  Loss {:.4f}  Time {:.4f} s'.format(
            epoch, i, len(dataloader), loss, time.clock()-start_time ))

def eval_op(net, dataloader, criterion, epoch, logger):

    start_time = time.clock()
    # Setting
    net.eval()

    # Evaluation
    n_cls = dataloader.dataset.CLASS_NUM
    with torch.no_grad():

        # loss AND N = [ [ TN, FP ], [ FN, TP ] ]
        loss = 0.0
        N = np.zeros((n_cls, n_cls))
        
        for i, (images_batch, labels_batch) in enumerate(tqdm(dataloader)):

            # forward
            labels_batch_pred = net(images_batch).cpu()
            labels_batch_pred = F.interpolate(labels_batch_pred, size=labels_batch.shape[2:4], mode='bilinear', align_corners=False)

            # loss
            loss += criterion(labels_batch_pred, labels_batch).item()

            # N = [ [ TN, FP ], [ FN, TP ] ]
            labels_batch_prob = F.softmax(labels_batch_pred, dim=1)[:,1:2,:,:]
            labels_batch_pred = torch.argmax(labels_batch_pred, dim=1)
            labels_batch_pred_np = labels_batch_pred.view(-1).numpy()
            
            labels_batch_np = labels_batch.view(-1).numpy()
            for idx in range(labels_batch_np.shape[0]):
                N[int(labels_batch_np[idx]), int(labels_batch_pred_np[idx])] += 1
        
            # qualitative comparison
            images_tensor = make_grid(images_batch, nrow=4, padding=12, normalize=True, range=(-1.0, 1.0), scale_each=False, pad_value=1.0)
            images_np = images_tensor.numpy().transpose((1,2,0))
            logger.log_images('Eval Images {}'.format(i), [images_np], step=epoch)

            labels_tensor = make_grid(labels_batch.unsqueeze(dim=1), nrow=4, padding=12, normalize=False, pad_value=1.0)
            labels_np = labels_tensor.numpy().transpose((1,2,0)).astype(np.float)
            logger.log_images('Eval GT Labels {}'.format(i), [labels_np], step=epoch)

            labels_pred_tensor = make_grid(labels_batch_pred.unsqueeze(dim=1), nrow=4, padding=12, normalize=False, pad_value=1.0)
            labels_pred_np = labels_pred_tensor.numpy().transpose((1,2,0)).astype(np.float)
            logger.log_images('Eval Predictive Labels {}'.format(i), [labels_pred_np], step=epoch)

            labels_prob_tensor = make_grid(labels_batch_prob, nrow=4, padding=12, normalize=False, pad_value=1.0)
            labels_prob_np = labels_prob_tensor.numpy().transpose((1,2,0))
            logger.log_images('Eval Predictive Prob {}'.format(i), [labels_prob_np], step=epoch)

        # Performance
        TN, FP, FN, TP = N[0,0], N[0,1], N[1,0], N[1,1]
        # Sensitivity & Specificity
        sensitivity = TP / (TP + FN)
        specificity = TN / (FP + TN)
        logger.log_value('sensitivity', sensitivity, step=i+epoch*len(dataloader))
        logger.log_value('specificity', specificity, step=i+epoch*len(dataloader))
        # IOU (Intersection Of Union) + mIOU
        IOU1 = TP / (FN + TP + FP)
        IOU0 = TN / (FN + TN + FP)
        mIOU = (IOU1 + IOU0) / 2.0
        logger.log_value('IOU1', IOU1, step=i+epoch*len(dataloader))
        logger.log_value('IOU0', IOU0, step=i+epoch*len(dataloader))
        logger.log_value('mIOU', mIOU, step=i+epoch*len(dataloader))
        # DSC (Dice Similarity Coefficient)
        DSC = 2*TP / ( 2*TP + FN + FP )
        logger.log_value('DSC', DSC, step=i+epoch*len(dataloader))
        # loss
        loss /= len(dataloader)
        logger.log_value('loss', loss, step=i+epoch*len(dataloader))

        print('Eval loss {:.4f} DSC {:.4f}  Time {:.4f} s'.format( 
            loss, DSC, time.clock()-start_time ))

        return DSC, IOU1, sensitivity, specificity
