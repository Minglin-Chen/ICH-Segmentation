import numpy as np
import cv2

def random_flip_pair(image, label):

    coin = np.random.randint(0, 2)
    if coin == 0:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    return image, label

def ramdom_rotation_pair(image, label):

    H, W = image.shape

    center = (W/2, H/2)
    angle = np.random.randint(0, 360)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)

    image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,  borderValue=(0))
    label = cv2.warpAffine(label, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE, borderValue=(0))

    return image, label