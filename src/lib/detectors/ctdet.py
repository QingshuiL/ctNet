from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 用于推理阶段
import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms
    from external.nms import nms
    from models.utils import cpu_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor

from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector
from .clu_base_detector import clu_BaseDetector

class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]  ### a 为一个字典 {hm reg wh} 256, 128, 64, 32
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2  # torch.flip()  按照3维度翻转图片
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets, cluster_dets = ctdet_decode(
                    hm, wh, reg=reg, 
                    cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)  # 256x256上的前K个det[boxs, sorces, class]
                # [1, K, 6]
        if return_time:
            return output, dets, cluster_dets, forward_time
        else:
            return output, dets, cluster_dets

    def post_process(self, dets_dict, clust_reg, clusted_dets, meta, scale=1):
        clust_ori_reg = None
        clust_ori_dets = None
        if self.opt.cluster > 0:
            if clust_reg is None:
                clust_ori_reg = None
            else:
                dets = clust_reg
                clust_ori_reg = ctdet_post_process(dets.copy(), [meta['c']],
                                                   [meta['s']],
                                                   meta['out_height'],
                                                   meta['out_width'],
                                                   self.opt.num_classes,
                                                   clust=True)
            if clusted_dets is None:
                clust_ori_dets = None
            else:
                dets = np.zeros(
                    [clusted_dets.shape[0], clusted_dets.shape[1], 4],
                    dtype=np.float32)
                dets[0, :, :] = clusted_dets[0, :, 1:5]
                clust_ori_dets = clusted_dets
                clust_ori_dets_ = ctdet_post_process(
                    dets.copy(), [meta['c']],[meta['s']],
                    meta['out_height'],meta['out_width'],
                    self.opt.num_classes,clust=True)
                clust_ori_dets[0, :, 1:5] = clust_ori_dets_[0][0, :, :]
        # 处理原始目标框
        dets = dets_dict.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'],self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0], clust_ori_reg, clust_ori_dets

    def merge_outputs(self, detections):  # detections列表包含一个字典:以cat为key的字典
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections],axis=0).astype(np.float32)  # results为字典：{1：value,...}
            results[j] = results[j][np.lexsort(-results[j].T)]  # 按最后一列降序

            if self.opt.nms == 0:
                soft_nms(results[j], Nt=self.opt.nms_rate, method=2)  # soft_nms  同一位置目标默认选择顺序在前的目标
            elif self.opt.nms == 1:
                results[j] = results[j][nms(results[j], 0.5)]

        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])  # 竖直方向平铺,提取所有类别scores合并列向量
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            # thresh1 = np.partition(scores, kth)[kth]   #
            # thresh2 = 0.2
            # thresh = max([thresh1, thresh2])
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(
                        detection[i, k, :4],
                        detection[i, k, -1],
                        detection[i, k, 4],
                        img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4],
                                           j - 1,
                                           bbox[4],
                                           img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)



class Clu_Detector(clu_BaseDetector):
    def __init__(self, opt):
        super(Clu_Detector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            outputs = self.model(images)  ### a 为一个字典 {hm reg wh} 256, 128, 64, 32
            det_list = []
            for i in range(self.opt.num_stacks):
                output = outputs[i]
                hm = output['hm'].sigmoid_()
                wh = output['wh']
                reg = output['reg'] if self.opt.reg_offset else None
                if self.opt.flip_test:
                    hm = (hm[0:1] +
                          flip_tensor(hm[1:2])) / 2  # torch.flip()  按照3维度翻转图片
                    wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                    reg = reg[0:1] if reg is not None else None
                torch.cuda.synchronize()
                forward_time = time.time()
                dets, cluster_dets = ctdet_decode(
                    hm,
                    wh,
                    reg=reg,
                    cat_spec_wh=self.opt.cat_spec_wh,
                    K=self.opt.K)  # 256x256上的前K个det[boxs, sorces, class]
                # [1, K, 6]
                det_list.append(dets)
        if return_time:
            return output, det_list, cluster_dets, forward_time
        else:
            return output, det_list, cluster_dets

    def post_process(self, dets_dict, clust_reg, clusted_dets, meta, scale=1):
        clust_ori_reg = None
        clust_ori_dets = None
        if self.opt.gen_cluster:
            if clust_reg is None:
                clust_ori_reg = None
            else:
                dets = clust_reg
                clust_ori_reg = ctdet_post_process(dets.copy(), [meta['c']],
                                                   [meta['s']],
                                                   meta['out_height'],
                                                   meta['out_width'],
                                                   self.opt.num_classes,
                                                   clust=True)
            if clusted_dets is None:
                clust_ori_dets = None
            else:
                dets = np.zeros(
                    [clusted_dets.shape[0], clusted_dets.shape[1], 4],
                    dtype=np.float32)
                dets[0, :, :] = clusted_dets[0, :, 1:5]
                clust_ori_dets = clusted_dets
                clust_ori_dets_ = ctdet_post_process(dets.copy(), [meta['c']],
                                                     [meta['s']],
                                                     meta['out_height'],
                                                     meta['out_width'],
                                                     self.opt.num_classes,
                                                     clust=True)
                clust_ori_dets[0, :, 1:5] = clust_ori_dets_[0][0, :, :]
        dets = dets_dict[-1].detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],
                                  meta['out_height'], meta['out_width'],
                                  self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale

        return dets[0], clust_ori_reg, clust_ori_dets

    def merge_outputs(self, detections_lists):  # detections列表包含一个字典:以cat为key的字典
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections_lists],
                axis=0).astype(np.float32)  # results为字典：{1：value,...}
            results[j] = results[j][np.lexsort(-results[j].T)]  # 按最后一列降序

            if self.opt.nms == 0:
                soft_nms(results[j], Nt=self.opt.nms_rate,
                         method=2)  # soft_nms  同一位置目标默认选择顺序在前的目标
            elif self.opt.nms == 1:
                results[j] = results[j][nms(results[j], 0.5)]

        scores = np.hstack([
            results[j][:, 4] for j in range(1, self.num_classes + 1)
        ])  # 竖直方向平铺,提取所有类别scores合并列向量
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            # thresh1 = np.partition(scores, kth)[kth]   #
            # thresh2 = 0.2
            # thresh = max([thresh1, thresh2])
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]

        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(
                        detection[i, k, :4],
                        detection[i, k, -1],
                        detection[i, k, 4],
                        img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4],
                                           j - 1,
                                           bbox[4],
                                           img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)