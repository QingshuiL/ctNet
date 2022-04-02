

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import os
import _init_paths

annType =['bbox']
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))

#initialize COCO ground truth api
dataDir='../'
dataType='val2014'
annFile = 'C:/Users/10295/CV/CenterNet-master/data/visdrone2019/annotations/test.json'
cocoGt=COCO(annFile)

#initialize COCO detections api
#resFile = r'C:\Users\10295\CV\CenterNet-master\exp\ctdet\visdrone2019_hg\results.json'
resFile = r'C:\Users\10295\CV\CenterNet-master\exp\ctdet\vis19_hg\results.json'
cocoDt=cocoGt.loadRes(resFile)

import json
#imgIds=sorted(cocoGt.getImgIds())
#imgIds=imgIds[0:100]
#imgId = imgIds[np.random.randint(100)]
dts = json.load(open(resFile,'r'))
imgIds = [imid['image_id'] for imid in dts]
imgIds = sorted(list(set(imgIds)))
del dts

# running evaluation

print(cocoGt.loadCats(cocoGt.getCatIds()))
cocoEval = COCOeval(cocoGt,cocoDt,"bbox")
cocoEval.params.imgIds  = imgIds
# 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
cats = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print(cats)
cocoEval.params.catIds = cats
cocoEval.params.maxDets = [1, 20, 100, 400]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()























