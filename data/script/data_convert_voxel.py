import os
import random
import math
import shutil
import cv2
import numpy as np

from mhd_loader import raw_loader

CLS_LIST = ['Huge', 'Large', 'Little', 'Median', 'Normal']

def split_train_val(dataset_root, K_fold=5):
	# 1. Load filenames
	image_names = os.listdir(os.path.join(dataset_root, 'images'))

	# 2. Split
	random.shuffle(image_names)
	N = math.floor( len(image_names) / 5 )

	images_K = []
	for i in range(K_fold-1):
		images_K.append( image_names[i*N:(i+1)*N] )
	images_K.append( image_names[(K_fold-1)*N:] )

	for i, item in enumerate(images_K):

		images_val = item
		images_train = []
		for j in range(K_fold):
			if i != j:
				images_train += images_K[j]

		labels_val_path = [ 'labels/'+image_val[0:-9]+'ICH.png' for image_val in images_val ]
		labels_train_path = [ 'labels/'+image_train[0:-9]+'ICH.png' for image_train in images_train ]

		images_val_path = [ 'images/'+image_val for image_val in images_val ]
		images_train_path = [ 'images/'+image_train for image_train in images_train ]

		val_list = [ "{} {}\n".format(images_val_path[i], labels_val_path[i]) for i in range(len(labels_val_path)) ]
		train_list = [ "{} {}\n".format(images_train_path[i], labels_train_path[i]) for i in range(len(labels_train_path)) ]
		val_list[-1] = val_list[-1][0:-1]
		train_list[-1] = train_list[-1][0:-1]

		# 3. Write
		with open(os.path.join(dataset_root, 'val_tumor_fold_{}.txt'.format(i+1)), 'w') as f:
			f.writelines(val_list)
		with open(os.path.join(dataset_root, 'train_tumor_fold_{}.txt'.format(i+1)), 'w') as f:
			f.writelines(train_list)

def data_convert(dataset_root, sub_folder='Original', dst_folder='Processed', min_area=1):

	root_path = os.path.join(dataset_root, sub_folder)
	dst_path = os.path.join(dataset_root, dst_folder)
	image_dst_path = os.path.join(dst_path, 'images')
	label_dst_path = os.path.join(dst_path, 'labels')

	if os.path.exists(dst_path):
		shutil.rmtree(dst_path)
	os.makedirs(image_dst_path)
	os.makedirs(label_dst_path)
	
	for cls_name in CLS_LIST:
		cls_path = os.path.join(root_path, cls_name)
		instances = os.listdir(cls_path)
		for instance in instances:
			brain_path = os.path.join(cls_path, instance, 'Brain.raw')
			label_path = os.path.join(cls_path, instance, 'originalICH.raw')

			brain_images = raw_loader(brain_path, 256, 256, None, np.uint8)
			label_images = raw_loader(label_path, 256, 256, None, np.uint8)
			assert brain_images.shape[0] == label_images.shape[0]

			for idx in range(brain_images.shape[0]):
				
				print('{} - {} - {}/{}'.format(cls_name, instance, idx+1, brain_images.shape[0]))

				if np.sum(label_images[idx] != 0) < min_area:
					continue

				brain_image_dst_path = os.path.join(image_dst_path, '{}_{}_brain.png'.format(instance, idx))
				label_image_dst_path = os.path.join(label_dst_path, '{}_{}_ICH.png'.format(instance, idx))

				cv2.imwrite(brain_image_dst_path, brain_images[idx])
				cv2.imwrite(label_image_dst_path, label_images[idx])

	split_train_val(dst_path, K_fold=5)

if __name__=='__main__':
	random.seed(666)

	dataset_root = '../'
	data_convert(dataset_root, sub_folder='Original', dst_folder='Processed', min_area=1)
