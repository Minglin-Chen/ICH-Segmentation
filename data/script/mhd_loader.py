import SimpleITK as sitk
import numpy as np
import cv2

def mhd_loader(PATH):
	images_sitk = sitk.ReadImage(PATH)
	images_np = sitk.GetArrayFromImage(images_sitk)
	return images_np

def raw_loader(PATH, height, width, channels=None, dtype=np.uint8):

	images = np.fromfile(PATH, dtype=dtype)
	
	if channels is None:
		Total_Pixel_Num = images.shape[0]
		Image_Pixel_Num = height * width
		assert  Total_Pixel_Num % Image_Pixel_Num == 0
		channels = Total_Pixel_Num // Image_Pixel_Num
		
	images = images.reshape(channels, height, width)
			
	return images
	
def _test_mhd_loader():

	# 1. Load images
    PATH = 'exmaple_data/Brain.mhd'
    images_np = mhd_loader(PATH)
	
	# 2. Info
    N, height, width = images_np.shape
	
	# 3. Process
    max_value, min_value = images_np.max(), images_np.min()
    images_np = (images_np - min_value) / (max_value - min_value)
 
	# 3. Visulazation
    for i in range(N):
        print('{} / {}'.format(i+1, N))
        image = images_np[i]
        cv2.imshow('MHD IMAGE', image)
        cv2.waitKey()

def _test_raw_loader():

	# 1. Load images
    # PATH = 'exmaple_data/CT131261_JHM_T_512x512x11_0.42x0.42x10_20060101000003.raw'
    # images_np = raw_loader(PATH, 512, 512, None, np.short)
    PATH = 'exmaple_data/Brain.raw'
    images_np = raw_loader(PATH, 256, 256, None, np.uint8)
	
	# 2. Info
    N, height, width = images_np.shape
	
	# 3. Process
    max_value, min_value = images_np.max(), images_np.min()
    images_np = (images_np - min_value) / (max_value - min_value)
	
	# 3. Info
    for i in range(N):
        print('{} / {}'.format(i+1, N))
        image = images_np[i]
        cv2.imshow('RAW IMAGE', image)
        cv2.waitKey()
		
if __name__ == '__main__':
	# _test_mhd_loader()
	_test_raw_loader()
	
		