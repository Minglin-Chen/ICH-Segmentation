import os
import cv2
import numpy as np

def class_frequency(dataset_root):

    items = []
    with open(os.path.join(dataset_root, 'train_tumor_fold_1.txt'), 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        image_path, label_path = line.rstrip('\n').split(' ')
        items.append(label_path)

    Num_Background, Num_Target = 0, 0
    for item in items:
        label = cv2.imread(os.path.join(dataset_root, item))
        label = label[:,:,0]
        label[label!=0] = 1
        Num_Background += np.sum(label==0)
        Num_Target += np.sum(label!=0)
    
    print('Num_Background {} / Num_Target {} = {}'.format(Num_Background, Num_Target, Num_Background/Num_Target))

if __name__=='__main__':
    dataset_root = '../Processed'
    class_frequency(dataset_root)