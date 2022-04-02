import torch
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import Bar
from PIL import Image
import torchvision.transforms.functional as F

PIL_RESIZE_MODE = {'bilinear': Image.BILINEAR, 'nearest': Image.NEAREST}


def max_iou(regions, b):
    if len(regions)==0:
        return 0
    # get area of a
    maxiou = 0
    for a in regions:
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        # get area of b
        area_b = (b[2] - b[0]) * (b[3] - b[1])

        # get left top x of IoU
        iou_x1 = np.maximum(a[0], b[0])
        # get left top y of IoU
        iou_y1 = np.maximum(a[1], b[1])
        # get right bottom of IoU
        iou_x2 = np.minimum(a[2], b[2])
        # get right bottom of IoU
        iou_y2 = np.minimum(a[3], b[3])

        # get width of IoU
        iou_w = iou_x2 - iou_x1
        # get height of IoU
        iou_h = iou_y2 - iou_y1

        # get area of IoU
        area_iou = iou_w * iou_h
        # get overlap ratio between IoU and all area
        iou = area_iou / (area_a + area_b - area_iou)
        if iou>maxiou:
            maxiou = iou
    return maxiou

def crop_bbox(img_size, clust_reg, Crop_N=2, L=(512,512), th=0.5):
    # get shape
    W, H = img_size
    # each crop
    crop_regions = []
    num_regions = 0
    while num_regions<Crop_N:
        x1 = np.random.randint(W - L[0])
        # get left top y of crop bounding box
        y1 = np.random.randint(H - L[1])
            # get right bottom x of crop bounding box
        x2 = x1 + L[0]
            # get right bottom y of crop bounding box
        y2 = y1 + L[1]
        crop_region = np.array((x1, y1, x2, y2))
        maxiou = max_iou(crop_regions,crop_region)
        if maxiou <= th:
            crop_regions.append(crop_region)
            num_regions += 1

    return crop_regions

def crop_bbox2(img_size, clust_reg, Crop_N=2, L=(512,512), th=0.5):
    # get shape
    W, H = img_size
    # each crop
    if Crop_N==4:
        xyxy1 = np.array([0, 0, L[0], L[1]], dtype=np.float32)
        # xyxy2 = np.array([W-L[0], 0, W, L[1]], dtype=np.float32)
        # xyxy3 = np.array([0, H-L[1], L[0], H], dtype=np.float32)
        xyxy4 = np.array([W-L[0], H-L[1], W, H], dtype=np.float32)
        xyxy1[[0,2]] = xyxy1[[0,2]] + clust_reg[0]
        # xyxy2[[0,2]] = xyxy2[[0,2]] + clust_reg[0]
        # xyxy3[[0,2]] = xyxy3[[0,2]] + clust_reg[0]
        xyxy4[[0,2]] = xyxy4[[0,2]] + clust_reg[0]

        xyxy1[[1,3]] = xyxy1[[1,3]] + clust_reg[1]
        # xyxy2[[1,3]] = xyxy2[[1,3]] + clust_reg[1]
        # xyxy3[[1,3]] = xyxy3[[1,3]] + clust_reg[1]
        xyxy4[[1,3]] + xyxy4[[1,3]] + clust_reg[1]
        # crop_regions = [xyxy1, xyxy2, xyxy3, xyxy4]
        crop_regions = [xyxy1, xyxy4]
    if Crop_N==2:
        xyxy1 = np.array([0, 0, L[0], L[1]], dtype=np.float32)
        xyxy2 = np.array([W-L[0], H-L[1], W, H], dtype=np.float32)
        xyxy1[[0,2]], xyxy2[[0,2]] = xyxy1[[0,2]] + clust_reg[0], xyxy2[[0,2]] + clust_reg[0]
        xyxy1[[1,3]], xyxy2[[1,3]] = xyxy1[[1,3]] + clust_reg[1], xyxy2[[1,3]] + clust_reg[1]
        crop_regions = [xyxy1, xyxy2]
    return crop_regions

class transform_det_scale_512(object):
    def __init__(self,             
                 mu=79.5,     
                 sigma=5,
                 scale_range=(0.5, 2.), 
                 default_scale=1.0,  
                 out_scale_deal='clip', 
                 mode='bilinear',  
                 debug_close_record=True):
        self.scale_range = scale_range
        self.mu = mu
        self.sigma = sigma
        self.default_scale = default_scale
        self.mode = PIL_RESIZE_MODE[mode]
        self.scale_range = scale_range   # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal
    
    def _sample_scale(self, src_size):
        nol_size = np.random.normal(self.mu, self.sigma)
        scale = nol_size / src_size
        return scale, nol_size

    def default_scale_deal(self, image, target):
        scale = self.default_scale
        # resize bbox mean size to our want size
        size = int(round(scale * image.height)), int(round(scale * image.width))
        target = target.resize((size[1], size[0]))
        image = F.resize(image, size, self.mode)
        return image, target, scale

    def just_clust_region(self, new_clu_size, clust_reg, image_wh):
        new_clust_reg = clust_reg.copy()
        w_dis = new_clu_size[0] - (new_clust_reg[2] - new_clust_reg[0] + 1)
        h_dis = new_clu_size[1] - (new_clust_reg[3] - new_clust_reg[1] + 1)
        new_clust_reg[0] = new_clust_reg[0] - w_dis/2
        new_clust_reg[1] = new_clust_reg[1] - h_dis/2
        new_clust_reg[2] = new_clust_reg[2] + w_dis/2
        new_clust_reg[3] = new_clust_reg[3] + h_dis/2
        if new_clust_reg[0] < 0 :
            if new_clust_reg[2] + (0 - new_clust_reg[0]) <= image_wh[0]:
                print('----------往右移动----------')
                new_clust_reg[[0,2]] = new_clust_reg[[0,2]] + (0 - new_clust_reg[0])
            else:
                print('----------往右有限移动----------')
                move_dis = image_wh[0] - new_clust_reg[2]
                if move_dis>=0:
                    move_dis=0
                new_clust_reg[[0,2]] = new_clust_reg[[0,2]] + move_dis
                new_clust_reg[0] = np.maximum(new_clust_reg[0], 0)

        if new_clust_reg[1] < 0 :
            if new_clust_reg[3] + (0 - new_clust_reg[1]) <= image_wh[1]:
                print('----------往下移动----------')
                new_clust_reg[[1,3]] = new_clust_reg[[1,3]] + (0 - new_clust_reg[1])
            else:
                print('----------往下有限移动----------')
                move_dis = image_wh[1] - new_clust_reg[3]  
                if move_dis<=0:
                    move_dis=0
                new_clust_reg[[1,3]] = new_clust_reg[[1,3]] + move_dis
                new_clust_reg[1] = np.maximum(new_clust_reg[1], 0)

        w_r_gap = (new_clust_reg[2] - image_wh[0])
        if  w_r_gap > 0:
            if new_clust_reg[0] - w_r_gap >= 0:
                print('----------往左移动----------')
                new_clust_reg[[0,2]] = new_clust_reg[[0,2]] - w_r_gap
            else:
                print('----------往左有限移动----------')
                move_dis = new_clust_reg[0] - 0
                if move_dis<=0:
                    move_dis=0
                new_clust_reg[[0,2]] = new_clust_reg[[0,2]] - move_dis
                new_clust_reg[2] = np.minimum(new_clust_reg[2], image_wh[0])
        
        h_b_gap = (new_clust_reg[3] - image_wh[1])
        if h_b_gap > 0:
            if new_clust_reg[1] - h_b_gap >=0 :
                print('----------往上移动----------')
                new_clust_reg[[1,3]] = new_clust_reg[[1,3]] - h_b_gap
            else:
                print('----------往上有限移动----------')
                move_dis = new_clust_reg[1] - 0
                if move_dis<=0:
                    move_dis=0
                new_clust_reg[[1,3]] = new_clust_reg[[1,3]] - move_dis
                new_clust_reg[3] = np.minimum(new_clust_reg[3], image_wh[1])
                
        return [new_clust_reg]

    def cut_extend_clust_region(self, scale, reg_size, clust_reg, image_wh):
        '''
        调整未缩放前的聚类区域，使得它乘上scale后尺寸在512附近
        '''
        w = reg_size[0]
        h = reg_size[1]
        clust_reg_size = clust_reg[2] - clust_reg[0] + 1, clust_reg[3] - clust_reg[1] + 1
        # 不足的扩展，超了的随机裁剪
        if w < 512 and h < 512:
            new_clu_size = 512/scale, 512/scale
            new_regions = self.just_clust_region(new_clu_size, clust_reg, image_wh)
        elif h<512 and 512< w and w<612:
            new_clu_size = w/scale, 512/scale
            new_regions = self.just_clust_region(new_clu_size, clust_reg, image_wh)
        elif h>512 and h<612 and w<512:
            new_clu_size = 512/scale, h/scale
            new_regions = self.just_clust_region(new_clu_size, clust_reg, image_wh)
        elif h>512 and h<612 and w>512 and w<612:
            new_regions = [clust_reg]
        elif h>612 and h<824 and w>612 and w<824:
            new_clu_size = 512/scale, 512/scale
            new_regions = crop_bbox2(clust_reg_size, clust_reg, Crop_N=2, L=new_clu_size, th=0.5)
        else:
            new_clu_size = 512/scale, 512/scale
            new_regions = crop_bbox2(clust_reg_size, clust_reg, Crop_N=4, L=new_clu_size, th=0.5)
        
        return new_regions

    def __call__(self, clust_regs, detections, image_wh):
        # print(len(target.bbox))
        # if len(detections) == 0:
        #     return self.default_scale_deal(image, detections)
        assert len(detections) != 0,'the length of detections is 0'
        clust_regs = clust_regs.reshape(clust_regs.shape[1], clust_regs.shape[2])
        detections = detections.reshape(detections.shape[1], detections.shape[2])
        # record old target info
        old_dets = detections.copy()
        num_clust = list(np.unique(detections[:,0]).astype(int))
        # cal mean size of image's bbox
        new_regions = []
        scales = []
        
        for ind, i in enumerate(num_clust):
            clust_reg = clust_regs[ind,:]
            dets = detections[detections[:,0] == i]
            bboxs_wh = np.array([dets[:,3] - dets[:,1] + 1, dets[:,4] - dets[:,2] + 1])
            bboxs_wh = bboxs_wh.reshape(bboxs_wh.shape[1], 2)
            bboxs_wh = np.maximum(bboxs_wh, 0)
            sizes = np.sqrt(bboxs_wh[:, 0] * bboxs_wh[:, 1])
            src_size = sizes.mean()
            scale, dst_size = self._sample_scale(src_size)
            if self.out_scale_deal == 'clip':
                if scale >= self.scale_range[1]:
                    scale = self.scale_range[1]     # note 1
                elif scale <= self.scale_range[0]:
                    scale = self.scale_range[0]   # note 1

            # resize bbox mean size to our want size
            clu_w = clust_reg[2] - clust_reg[0] + 1
            clu_h = clust_reg[3] - clust_reg[1] + 1
            init_new_size = int(round(scale * clu_w)), int(round(scale * clu_h))

            new_region = self.cut_extend_clust_region(scale, init_new_size, clust_reg, image_wh)
            new_regions.append(new_region)
            scales.append(scale)

        return new_regions, scales


class transform_det_scale_1024(object):
    def __init__(self,             
                 mu=79.5,     
                 sigma=5,
                 scale_range=(0.5, 2.), 
                 default_scale=1.0,  
                 out_scale_deal='clip', 
                 mode='bilinear',  
                 debug_close_record=True):
        self.scale_range = scale_range
        self.mu = mu
        self.sigma = sigma
        self.default_scale = default_scale
        self.mode = PIL_RESIZE_MODE[mode]
        self.scale_range = scale_range   # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal

    def _sample_scale(self, src_size):
        nol_size = np.random.normal(self.mu, self.sigma)
        scale = nol_size / src_size
        return scale, nol_size


    def default_scale_deal(self, image, target):
        scale = self.default_scale
        # resize bbox mean size to our want size
        size = int(round(scale * image.height)), int(round(scale * image.width))
        target = target.resize((size[1], size[0]))
        image = F.resize(image, size, self.mode)
        return image, target, scale


    def just_clust_region(self, new_clu_size, clust_reg, image_wh):
        new_clust_reg = clust_reg.copy()
        w_dis = new_clu_size[0] - (new_clust_reg[2] - new_clust_reg[0] + 1)
        h_dis = new_clu_size[1] - (new_clust_reg[3] - new_clust_reg[1] + 1)
        new_clust_reg[0] -= w_dis/2
        new_clust_reg[1] -= h_dis/2
        new_clust_reg[2] += w_dis/2
        new_clust_reg[3] += h_dis/2
        if new_clust_reg[0] < 0 :
            if new_clust_reg[2] + (0 - new_clust_reg[0]) <= image_wh[0]:
                print('----------往右移动----------')
                new_clust_reg[[0,2]] = new_clust_reg[[0,2]] + (0 - new_clust_reg[0])
            else:
                print('----------往右有限移动----------')
                move_dis = image_wh[0] - new_clust_reg[2]
                if move_dis<=0:
                    move_dis=0
                new_clust_reg[[0,2]] = new_clust_reg[[0,2]] + move_dis
                new_clust_reg[0] = np.maximum(new_clust_reg[0], 0)

        if new_clust_reg[1] < 0 :
            if new_clust_reg[3] + (0 - new_clust_reg[1]) <= image_wh[1]:
                print('----------往下移动----------')
                new_clust_reg[[1,3]] = new_clust_reg[[1,3]] + (0 - new_clust_reg[1])
            else:
                print('----------往下有限移动----------')
                move_dis = image_wh[1] - new_clust_reg[3]  
                if move_dis<=0:
                    move_dis = 0
                new_clust_reg[[1,3]] = new_clust_reg[[1,3]] + move_dis
                new_clust_reg[1] = np.maximum(new_clust_reg[1], 0)

        w_r_gap = (new_clust_reg[2] - image_wh[0])
        if  w_r_gap > 0:
            if new_clust_reg[0] - w_r_gap >= 0:
                print('----------往左移动----------')
                new_clust_reg[[0,2]] = new_clust_reg[[0,2]] - w_r_gap
            else:
                print('----------往左有限移动----------')
                move_dis = new_clust_reg[0] - 0
                if move_dis<=0:
                    move_dis = 0
                new_clust_reg[[0,2]] = new_clust_reg[[0,2]] - move_dis
                new_clust_reg[2] = np.minimum(new_clust_reg[2], image_wh[0])
        
        h_b_gap = (new_clust_reg[3] - image_wh[1])
        if h_b_gap > 0:
            if new_clust_reg[1] - h_b_gap >=0 :
                print('----------往上移动----------')
                new_clust_reg[[1,3]] = new_clust_reg[[1,3]] - h_b_gap
            else:
                print('----------往上有限移动----------')
                move_dis = new_clust_reg[1] - 0
                if move_dis<=0:
                    move_dis = 0
                new_clust_reg[[1,3]] = new_clust_reg[[1,3]] - move_dis
                new_clust_reg[3] = np.minimum(new_clust_reg[3], image_wh[1])
                
        return [new_clust_reg]

    def cut_extend_clust_region(self, scale, reg_size, clust_reg, image_wh):
        '''
        调整未缩放前的聚类区域，使得它乘上scale后尺寸在512附近
        '''
        w = reg_size[0]
        h = reg_size[1]
        clust_reg_size = clust_reg[2] - clust_reg[0] + 1, clust_reg[3] - clust_reg[1] + 1
        # 不足的扩展，超了的随机裁剪
        if w < 1024 and h < 1024:
            new_clu_size = 1024/scale, 1024/scale
            new_regions = self.just_clust_region(new_clu_size, clust_reg, image_wh)
        elif h<1024 and 1024< w and w<1224:
            new_clu_size = w/scale, 1024/scale
            new_regions = self.just_clust_region(new_clu_size, clust_reg, image_wh)
        elif h>1024 and h<1224 and w<1024:
            new_clu_size = 1024/scale, h/scale
            new_regions = self.just_clust_region(new_clu_size, clust_reg, image_wh)
        elif h>1024 and h<1224 and w>1024 and w<1224:
            new_regions = [clust_reg]
        elif h>1224 and h<1714 and w>1224 and w<1714:
            new_clu_size = 1024/scale, 1024/scale
            new_regions = crop_bbox2(clust_reg_size, clust_reg, Crop_N=2, L=new_clu_size, th=0.5)
        else:
            new_clu_size = 1024/scale, 1024/scale
            new_regions = crop_bbox2(clust_reg_size, clust_reg, Crop_N=4, L=new_clu_size, th=0.5)
        
        return new_regions

    def __call__(self, clust_regs, detections, image_wh):
        # print(len(target.bbox))
        # if len(detections) == 0:
        #     return self.default_scale_deal(image, detections)
        assert len(detections) != 0,'the length of detections is 0'
        clust_regs = clust_regs.reshape(clust_regs.shape[1], clust_regs.shape[2])
        detections = detections.reshape(detections.shape[1], detections.shape[2])
        # record old target info
        old_dets = detections.copy()
        num_clust = list(np.unique(detections[:,0]).astype(int))
        # cal mean size of image's bbox
        new_regions = []
        scales = []
        
        for ind, i in enumerate(num_clust):
            clust_reg = clust_regs[ind,:]
            dets = detections[detections[:,0] == i]
            bboxs_wh = np.array([dets[:,3] - dets[:,1] + 1, dets[:,4] - dets[:,2] + 1])
            bboxs_wh = bboxs_wh.reshape(bboxs_wh.shape[1], 2)
            bboxs_wh = np.maximum(bboxs_wh, 0)
            sizes = np.sqrt(bboxs_wh[:, 0] * bboxs_wh[:, 1])
            src_size = sizes.mean()
            scale, dst_size = self._sample_scale(src_size)
            if self.out_scale_deal == 'clip':
                if scale >= self.scale_range[1]:
                    scale = self.scale_range[1]     # note 1
                elif scale <= self.scale_range[0]:
                    scale = self.scale_range[0]   # note 1

            # resize bbox mean size to our want size
            clu_w = clust_reg[2] - clust_reg[0] + 1
            clu_h = clust_reg[3] - clust_reg[1] + 1
            init_new_size = int(round(scale * clu_w)), int(round(scale * clu_h))

            new_region = self.cut_extend_clust_region(scale, init_new_size, clust_reg, image_wh)
            new_regions.append(new_region)
            scales.append(scale)

        return new_regions, scales





