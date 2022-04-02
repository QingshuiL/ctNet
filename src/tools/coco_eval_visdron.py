
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json

def spe_cat_eval(dataType, resFile, cat_ids):
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    print('Running demo for *%s* results.'%(annType))

    #initialize COCO ground truth api
    dataDir ='C:/Users/10295/CV/CenterNet-master/data/visdrone2019'
    # dataType ='train'
    annFile = '%s/annotations/%s.json'%(dataDir,dataType)
    cocoGt = COCO(annFile)

    #initialize COCO detections api
    # resFile = 'C:/Users/10295/CV/CenterNet-master/exp/ctdet/visdrone2019_resfpn/test/results.json'
    cocoDt = cocoGt.loadRes(resFile)
    print(cocoDt.cats)
    '''
    imgIds=sorted(cocoGt.getImgIds())
    imgIds=imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]
    '''
    dts = json.load(open(resFile,'r'))
    imgIds = [imid['image_id'] for imid in dts]
    imgIds = sorted(list(set(imgIds)))
    del dts

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.params.catIds = cat_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    # train val test
    dataType = 'val'
    resFile = 'C:/Users/10295/CV/CenterNet-master/exp/ctdet/visdrone2019_dla34/results.json'
    # 1 2 3 4 5 6 7 8 9 10 11 12 range(1,13)
    cat_ids = [range(1,13)]
    spe_cat_eval(dataType, resFile, cat_ids)