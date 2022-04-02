import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch

data_dir = 'C:/Users/10295/CV/CenterNet-master/data/visdrone2019/'
img_dir = os.path.join(data_dir, 'images')
split = 'train'
annot_path = os.path.join(data_dir, 'annotations', split + '.json')
coco = coco.COCO(annot_path)
images = coco.getImgIds()
num_samples = len(images)
print('==> initializing visdrone2019 {} data.'.format(split))
print('Loaded {} {} samples'.format(split, num_samples))


def _to_float(self, x):
  return float("{:.2f}".format(x))

def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections


def save_results(self, results, save_dir):
  json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    cats = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print(cats)
    coco_eval.params.catIds = cats
    coco_eval.params.maxDets = [1, 20, 100, 400]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval.summarize_2()