from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import pycocotools.coco as coco

from utils.image import get_affine_transform
from utils.image import transform_preds

save_path = r'C:\Users\10295\CV\CenterNet-master\data\vis19_train_1024'
data_dir = 'C:/Users/10295/CV/CenterNet-master/data/visdrone2019'
annot_path = os.path.join(data_dir, 'annotations', 'train.json')
img_dir = os.path.join(data_dir, 'image')
vis19 = coco.COCO(annot_path)
Anns = vis19.anns                       # 353550 x 8:
                                        # ares bbox category_id id ignore iamge_id iscrowd segmentationg


imgIDs = vis19.getImgIds()
catids = {'0':2,'1':3,'2':4,'3': 5,'4': 6,'5': 7,'6': 8,'7':9,'8': 10,'9': 11}
areas = {}
areas_tot = np.zeros((10))
dets = []

for j in catids: # 0 ~ 9
    ann_ids = vis19.getAnnIds(catIds=[catids[j]])
    bbox = np.concatenate( [[Anns[k]['bbox']] for k in ann_ids], axis=0)
    cat = np.concatenate( [[Anns[k]['category_id']] for k in ann_ids], axis=0)
    areas[j] = bbox[:,2] * bbox[:,3]
    areas_tot[int(j)] = sum(areas[j])
    dets.append(np.c_[bbox, cat, areas[j]])

# a = list(areas_tot.values())
# areas_rate = []
areas_sum = sum((areas_tot))
areas_rate = areas_tot / areas_sum
print("areas rate:", areas_rate)

def pre_process(image, scale = 1, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)  # 缩放后的 高和宽
    new_width  = int(width * scale)
    inp_height, inp_width = 1024, 1024     # 1024x1024
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)   # 放缩后的中心点
    s = max(height, width) * 1.0   # 原最大边

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])  # 获得仿射变换矩阵，可有 平移 翻转 旋转 缩放 仿射变换
    resized_image = cv2.resize(image, (new_width, new_height))  # 放缩图像
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),      # 仿射变换  1024x1024x3
      flags=cv2.INTER_LINEAR)
    cv2.imwrite(save_path + vis19['imgs'][imIds], img)
    # cv2.imwrite("C:/Users/10295/Desktop/目标检测/Picture/image.jpg", image)
    # cv2.imwrite("C:/Users/10295/Desktop/目标检测/Picture/inp_image.jpg", inp_image)
    #inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)    # 标准化
    #images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 改变图片矩阵: 1x3x1024x1024
    meta = {'c': c, 's': s, 
            'out_height': inp_height ,   # 256
            'out_width': inp_width }  # 256           
    return inp_image, meta

def ctdet_post_process(dets, c, s, h, w, num_classes):
  ret = []
  for i in range(dets.shape[0]): # dets.shape[0] 表示图的数量
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))   # 坐标点反变换
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]

    for j in range(num_classes):
      inds = (classes == j)     # 按类别选出索引
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist() # 筛选类别为 j 的 x1,y1,x2,y2,scroes
    ret.append(top_preds)
  return ret


