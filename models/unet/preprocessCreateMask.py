import os
import random
import numpy as np

from tqdm import tqdm

from skimage.io import imread, imsave, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

print('Preprocessin Mask')

TRAINING_PATH = 'D:\GoogleDrive\Arbeit\HEROHE_Challenge\data-science-bowl-2018\stage1_train'
IMG_CHANNELS = 3

train_ids = next(os.walk(TRAINING_PATH))[1]

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAINING_PATH + '\\' + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask = np.maximum(mask, mask_)
    if not os.path.exists(path + '/mask'):
        os.makedirs(path + '/mask')
    imsave(path + '/mask/'+ id_ + '.png', mask)

print('Preprocessin Mask done')