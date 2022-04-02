from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import Bar
from PIL import Image
from torchvision import transforms

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
from cluster.sk_cluster import meanshift
from cluster.vis_cut_clust import vis_clust, cut_clust_gen_Dets

try:
    from external.nms import soft_nms
    from external.nms import nms
    from models.utils import cpu_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')

from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.post_process import ctdet_post_process
from cluster.transform_cluster import transform_det_scale_1024, transform_det_scale_512
from cluster.random_or_average_crop import random_crop, average_crop
from cluster.transform.transform_scale import GaussTransfrom
from cluster.transform.transform_data_1024 import cutfill_1024
from cluster.transform.transform_data import cutfill_512
from cluster.transform.BoxList import BoxList

class CtdetDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model_stage1 = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model_stage1 = load_model(self.model_stage1, opt.load_model_stage1)
        self.model_stage1 = self.model_stage1.to(opt.device)
        self.model_stage1.eval()
        if opt.cluster == 2:
            self.model_stage2 = create_model(opt.arch, opt.heads, opt.head_conv)
            self.model_stage2 = load_model(self.model_stage2, opt.load_model_stage2)
            self.model_stage2 = self.model_stage2.to(opt.device)
            self.model_stage2.eval()

        print('stage_input:',opt.stage_input)
        if opt.stage_input == 1024:
            self.Gauss_transform = GaussTransfrom(mu=98, sigma=5, scale_range=(0., 3.))
        elif opt.stage_input == 512:
            self.Gauss_transform = GaussTransfrom(mu=79.5, sigma=5, scale_range=(0.5, 3.))
        self.reg_transform = transform_det_scale_1024(mu=98, sigma=5, scale_range=(0.5, 2.), default_scale=1.0, out_scale_deal='clip')
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 1500  # 每张图片 每个类别NMS后限制目标框的最大目标数量
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True
    # 图片输入网络前的处理
    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)                                            # 缩放后的 高和宽
        new_width = int(width * scale)

        if self.opt.fix_res:                                                        # true
            inp_height, inp_width = self.opt.input_h, self.opt.input_w              # 1024x1024 512x512
            c = np.array([new_width / 2., new_height / 2.],
                         dtype=np.float32)                                          # 放缩后的中心点
            s = max(height, width) * 1.0                                            # 原最大边
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
            
        trans_input = get_affine_transform(
            c, s, 0, [inp_width, inp_height])                                       # 获得仿射变换矩阵，可有 平移 翻转 旋转 缩放 仿射变换
        resized_image = cv2.resize(image, (new_width, new_height))                  # 放缩图像
        inp_image = cv2.warpAffine(
            resized_image,
            trans_input,
            (inp_width, inp_height),                                                # 仿射变换  1024x1024x3
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(             # 标准化
            np.float32)  
        images = inp_image.transpose(2, 0, 1).reshape(                              # 改变图片的维度: 1x3x1024x1024
            1, 3, inp_height, inp_width) 
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]),                # RGB到BGR的转化并叠加两图 2x3x1024x1024
                                    axis=0)  
        images = torch.from_numpy(images)                                           #变为torch
        meta = {'c': c,
                's': s,
                'out_height': inp_height // self.opt.down_ratio,  # 256
                'out_width': inp_width // self.opt.down_ratio
            }  # 256
        return images, meta

    def pre_process_2(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)                                            # 缩放后的 高和宽
        new_width = int(width * scale)

        if self.opt.fix_res:                                                        # true
            inp_height, inp_width = self.opt.input_h_2, self.opt.input_w_2              # 1024x1024 512x512
            c = np.array([new_width / 2., new_height / 2.],
                         dtype=np.float32)                                          # 放缩后的中心点
            s = max(height, width) * 1.0                                            # 原最大边
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
            
        trans_input = get_affine_transform(
            c, s, 0, [inp_width, inp_height])                                       # 获得仿射变换矩阵，可有 平移 翻转 旋转 缩放 仿射变换
        resized_image = cv2.resize(image, (new_width, new_height))                  # 放缩图像
        inp_image = cv2.warpAffine(
            resized_image,
            trans_input,
            (inp_width, inp_height),                                                # 仿射变换  1024x1024x3
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(             # 标准化
            np.float32)  
        images = inp_image.transpose(2, 0, 1).reshape(                              # 改变图片的维度: 1x3x1024x1024
            1, 3, inp_height, inp_width) 
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]),                # RGB到BGR的转化并叠加两图 2x3x1024x1024
                                    axis=0)  
        images = torch.from_numpy(images)                                           #变为torch
        meta = {'c': c,
                's': s,
                'out_height': inp_height // self.opt.down_ratio,  # 256
                'out_width': inp_width // self.opt.down_ratio
            }  # 256
        return images, meta


    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model_stage1(images)[-1]  ### a 为一个字典 {hm reg wh} 256, 128, 64, 32
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2  # torch.flip()  按照3维度翻转图片
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            # 由热力图筛选得到bbox
            dets = ctdet_decode(
                    hm, wh, reg=reg, 
                    cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)  # 256x256上的前K个det[boxs, sorces, class]
                # [1, K, 6]
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets


    def process_clu(self, images, return_time=False):
        with torch.no_grad():
            output = self.model_stage2(images)[-1]  ### a 为一个字典 {hm reg wh} 256, 128, 64, 32
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2  # torch.flip()  按照3维度翻转图片
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            # 由热力图筛选得到bbox
            dets = ctdet_decode(hm, wh, reg=reg,
                    cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)  # 256x256上的前K个det[boxs, sorces, class]
                # [1, K, 6]
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets
    
    # 将bbox映射回原始尺寸
    def post_process(self,  dets_dict, meta, scale=1):
        dets = dets_dict.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets, clust_ori_dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'],self.opt.num_classes)

        # clust_ori_dets = clust_ori_dets[clust_ori_dets[:,:,4]>0.15]
        # cx = (clust_ori_dets[:,0] + clust_ori_dets[:,2]) / 2
        # cy = (clust_ori_dets[:,1] + clust_ori_dets[:,3]) / 2
        # clust_ori_dets = np.concatenate((clust_ori_dets, cx.reshape(len(cx),1), cy.reshape(len(cy),1)), axis=1)    #bbox sorce class cx cy

        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale

        # return dets[0] ,clust_ori_dets
        return dets[0]

    def post_process_clu(self,  dets_dict, meta, scale=1):
        dets = dets_dict.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets, clust_dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'],self.opt.num_classes)

        clust_dets[:,:,:4] = clust_dets[:,:,:4] * 1/meta['reg_scale']
        clust_dets[:,:,[0,2]] = clust_dets[:,:,[0,2]] + meta['clu_coord'][0]
        clust_dets[:,:,[1,3]] = clust_dets[:,:,[1,3]] + meta['clu_coord'][1]
        
        clust_ori_dets = {}
        for i in range(clust_dets.shape[0]):
            classes = clust_dets[i, :, -1]
            for j in range(self.opt.num_classes):
                inds = (classes == j)     
                clust_ori_dets[j + 1] = np.concatenate([
                        clust_dets[0, inds, :4].astype(np.float32),
                        clust_dets[0, inds, 4:5].astype(np.float32)], axis=1).tolist()
        
        for j in range(1, self.num_classes + 1):
            clust_ori_dets[j] = np.array(clust_ori_dets[j], dtype=np.float32).reshape(-1, 5)
            clust_ori_dets[j][:, :4] /= scale

        # return dets[0] ,clust_ori_dets
        return clust_ori_dets

    # bbox聚合 nms
    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections],axis=0).astype(np.float32)  # results为字典：{1：value,...}
            results[j] = results[j][np.lexsort(-results[j].T)]  # 按最后一列降序
        
            # x_keep_inds = (results[j][:, 2] - results[j][:, 0] > 2)
            # results[j] = results[j][x_keep_inds]
            # x_keep_inds = (results[j][:, 3] - results[j][:, 1] > 2)
            # results[j] = results[j][x_keep_inds]

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

    def run(self, image_or_path_or_tensor, phase=None, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset,
                            ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        phase = phase
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):                     # ('') 为 str类型
            # image_or_path_or_tensor = r'I:\学习文件夹\cv\ctNet\data\visdrone2019\images\000187.jpg'
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        #
        detections_list = []
        dte_append = detections_list.append
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)          # image为原始图片;需执行 ctdet不用meta;
                                                                            # 本步得到 scale flip 后的图片
            else:                                                           # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                # 主要是将torch转化为numpy类型
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            # 图片转移到GPU
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(images, return_time=True)  # 通过网络推理
            # 256x256上的前K个det[boxs, sorces, class]
            torch.cuda.synchronize()  # 同步，加上GPU上的计算时间
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time



            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)

            dets_list = self.post_process(
                dets, meta, scale)  # 1x200x6的256x256的dets 还原到原图像去
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            dte_append(dets_list)
            
        results = self.merge_outputs(detections_list)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results)
        
        # 处理 detections 转化为numpy格式
        labels = []
        labels_append = labels.append
        for i,anns in results.items():
            labels_append(np.full((len(anns)), int(i)))
        labels = np.concatenate(labels)

        anns = np.concatenate( [bboxs for i,bboxs in results.items()], axis=0 ).astype(np.float32)
        clusted_ori_dets = np.concatenate((anns, labels.reshape(len(labels),1)), axis=1).astype(np.float32)

        if self.opt.cluster>0 and not self.opt.average_crop and not self.opt.random_crop:
            crop_reg, crop_dets = meanshift(cluster_dets=clusted_ori_dets)
        elif self.opt.cluster>0 and self.opt.average_crop:
            crop_reg, crop_dets = average_crop(image.shape[0:2])
        elif self.opt.cluster>0 and self.opt.random_crop:
            crop_reg, crop_dets = random_crop(image.shape[0:2])
        else:
            crop_reg, crop_dets = None, None
        
        if not (crop_reg is None or crop_dets is None):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.figure(11)
            # plt.clf()
            # plt.imshow(img)
            # plt.show()
            if not self.opt.average_crop and not self.opt.random_crop and not self.opt.not_transform:
                just_clu_regions, scales = self.reg_transform(crop_reg, crop_dets, (image.shape[1],image.shape[0]))
            elif self.opt.average_crop or self.opt.random_crop or self.opt.not_transform:
                just_clu_regions = crop_reg
                scales = [1] * len(crop_reg)
                
            # a = crop_reg[0][0,:4]
            # croped = img.crop(a)
            # plt.figure(12)
            # plt.clf()
            # plt.imshow(croped)
            # plt.show()

            crop_imgs = []
            reg_coordin = []
            reg_scales = []
            for i, regs in enumerate(just_clu_regions):
                scale = scales[i]
                for reg in regs:
                    crop_img = img.crop(reg[:4])
                    size = reg[2] - reg[0] + 1,  reg[3] - reg[1] + 1
                    size = int(size[0]*scale), int(size[1]*scale)
                    crop_img = crop_img.resize(size, Image.ANTIALIAS)

                    # plt.figure(1)
                    # plt.clf()
                    # plt.imshow(crop_img)
                    # plt.show()

                    crop_imgs.append(crop_img)
                    reg_coordin.append(reg)
                    reg_scales.append(scale)
            # gen_Dets()
        else:
            crop_imgs = None
            

        if self.opt.cluster == 2 and not crop_imgs is None:
            for i, clu_image in enumerate(crop_imgs):

                    # for j, im in enumerate(imgs):
                    #     show_bbox = bboxss[j]
                    #     plt.figure(j+1)
                    #     plt.imshow(im)
                    #     for k in range(len(show_bbox)):
                    #         plt.gca().add_patch(plt.Rectangle(xy=(show_bbox[k,0], show_bbox[k,1]),
                    #                             width=show_bbox[k,2] - show_bbox[k,0], 
                    #                             height=show_bbox[k,3] - show_bbox[k,1],
                    #                             edgecolor='r',
                    #                             fill=False, linewidth=2))
                    # plt.show()
                reg_scale = reg_scales[i]
                clu_img = cv2.cvtColor(np.asarray(clu_image), cv2.COLOR_RGB2BGR)
                clu_img, meta = self.pre_process_2(clu_img, scale=1.0, meta=None)
                meta['reg_scale'] = reg_scale
                meta['clu_coord'] = reg_coordin[i]
                clu_img = clu_img.to(self.opt.device)
                torch.cuda.synchronize()
                output, clu_dets_list = self.process_clu(clu_img, return_time=False) 
                clu_dets_list = self.post_process_clu(clu_dets_list, meta)
                dte_append(clu_dets_list)

            results = self.merge_outputs(detections_list)


        return {
            'results': results,
            'tot': tot_time,
            'load': load_time,
            'pre': pre_time,
            'net': net_time,
            'dec': dec_time,
            'post': post_time,
            'merge': merge_time
        }
