from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.ctdet_detector import CtdetDetector

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = CtdetDetector
  
  if opt.on_train:
    split = 'train'
  else:
    split = 'val' if not opt.trainval else 'test'

  if opt.cluster == 0:
    print('-----------Dont generate cluster----------')
  elif opt.cluster == 1:
    print('----------产生用于训练的聚类图片----------')
  elif opt.cluster == 2:
    print('----------最终测试阶段，分两阶段粗检测和精细检测----------')
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  # 加载图片时就预先对图片进行处理
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)


challenge_images_dir = 'D:/WangYi/cv/ctNet/data/visdrone2019/challenge_images'
challenge_images_list = os.listdir(challenge_images_dir)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = CtdetDetector

  if opt.on_train:
    split = 'train'
  else:
    split = 'test' if not opt.trainval else 'val'
    
  if opt.cluster == 0:
    print('-----------Dont generate cluster----------')
  elif opt.cluster == 1:
    print('----------产生用于训练的聚类图片----------')
  elif opt.cluster == 2:
    print('----------最终测试阶段，分两阶段粗检测和精细检测----------')
  # split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)  
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):   # 一张图片 产生det
    # if ind < 804:
    #   print("skip:",ind)
    #   continue
    if opt.challenge:
      img_name = challenge_images_list[ind]
      img_id = img_name[:-4]
      img_path = os.path.join(dataset.img_dir, img_name)
    else:
      img_id = dataset.images[ind]
      img_info = dataset.coco.loadImgs(ids=[img_id])[0]
      img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    ret = detector.run(img_path, phase=split)    ## 运行

    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()

#   if opt.inference_type == '0':
  dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)