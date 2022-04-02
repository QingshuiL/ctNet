from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .sample.ctdet import CTDetDataset
from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.coco_hp import COCOHP
from .dataset.visdrone2019 import Visdrone2019
from .dataset.uavdt import UAVDT

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'coco_hp': COCOHP,
  'visdrone2019': Visdrone2019,
  'uavdt': UAVDT,
}

_sample_factory = {
  'ctdet': CTDetDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
