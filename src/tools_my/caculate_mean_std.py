import numpy as np
import cv2
import random
import os,glob
from tqdm import tqdm

# calculate means and std
image_dir = "D:\DataSets\Dota\images"
image_list = os.listdir(image_dir)
print("图片的数量为:",len(image_list))
means, stdevs = [], []
for i,im_name in tqdm(enumerate(image_list)):
    img_path = os.path.join(image_dir,im_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_h, img_w = img.shape[0], img.shape[1]
    mean = np.array([img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()])
    std = np.array([img[:,:,0].std(), img[:,:,1].std(), img[:,:,2].std()])
    means.append(mean)
    stdevs.append(std)

print("得到的均值和方差的条数:",len(means))

total_means = np.vstack(means)
total_std = np.vstack(stdevs)
mean = np.mean(total_means / 255,axis=0)
std = np.mean(total_std / 255,axis=0)

  
print("normMean = {}".format(mean))
print("normStd = {}".format(std))
