from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import json
import cv2
import os
import math


def get_dir(src_point, rot_rad):  # 旋转变换
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,   # 变换后中心
                         scale,    # 变换后最大边
                         rot,      # 0
                         output_size,  # 1024, 1024 
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]      # 原最大边
    dst_w = output_size[0]    # 1024
    dst_h = output_size[1]    # 1024

    rot_rad = np.pi * rot / 180      # 旋转角度 默认为0
    src_dir = get_dir([0, src_w * -0.5], rot_rad)      # 
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift   # 对1360x765图片： 680x382.5 及中心点
    src[1, :] = center + src_dir + scale_tmp * shift #
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    # src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    # dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    src[2, :] = np.array([0, center[1]])
    dst[2, :] = np.array([0, 256])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))  # 包含变换：平移、选择、翻转、缩放、剪切
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[1], pt[0], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)    # 矩阵乘法或内积
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


class COCODataset(torchvision.datasets.coco.CocoCaptions):

    def __init__(self, ann_file, root):
        super(COCODataset, self).__init__(root, ann_file)
        self.img_ids = sorted(self.ids)
        print("valid image count in dataset: {}".format(len(self.img_ids)))

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], 
            dtype=np.float32)
        return bbox

    def __len__(self):
        length = len(self.coco.imgs)
        return length

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_inf = self.coco.loadImgs(ids=[img_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = len(anns)
        height, width = img_inf['height'], img_inf['width']
        c = np.array([height / 2., width / 2.], dtype=np.float32)
        s = max(width, height) * 1.0
        input_h, input_w = 1024, 1024
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            befor_bbox = np.copy(bbox)
            cls_id = int(ann['category_id'])
            bbox[:2] = affine_transform(bbox[:2], trans_input)
            bbox[2:] = affine_transform(bbox[2:], trans_input)
            # print('w{0} h{1} befor bbox{2}; after bbox{3}'.format(width, 
                # height, befor_bbox, bbox))
            test = np.array([0, 0, width, height])
            # print('befor:', test)
            # test[2:] = affine_transform(test[2:], trans_input)
            # test[:2] = affine_transform(test[:2], trans_input)
            # print('\n', test)
            # bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            # bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            gt_det.append([bbox, 1, cls_id])

        return gt_det


if __name__ == "__main__":
    # ann_file = 'C:/Users/10295/CV/ctNet/data/coco2017/annotations/instances_train2017.json'
    # ann_file = 'C:/Users/10295/CV/ctNet/data/coco2017/annotations/instances_train2017.json'
    ann_file = 'C:/Users/10295/CV/ctNet/data/visdrone2019/annotations/cluster_train.json'
    root = 'C:/Users/10295/CV/ctNet/data/coco2017/'
    data = COCODataset(ann_file, root)
    det_size = []

    for i, det in enumerate(data):
         for ann in det:
             bbox = ann[0]
             size = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
             det_size.append(size)
         print('deal with {0}/{1}'.format(i+1,len(data)))

    mean_size = np.mean(det_size)
    print('mean size of the dataset is :',mean_size)

# mean size of the dataset is : 79.50237   cluster_train.json 的目标的平均尺寸
# 1024 mean size of the dataset is : 97.95856   cluster_train.json 的目标的平均尺寸








