from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import pandas as pd



def post_clust_region(labels, cluster_dets, n_clusters): # 108x1 108x7 1
    clust = np.cat([labels, cluster_dets], dim=1)
    clu_list = []
    for i in range(n_clusters):
        cluster = clust[cluster[:,0]==i]
        clu_list.append(cluster)

    clust_reg = np.zeros([n_clusters,4]).astype(np.float32)

    for i in range(n_clusters):
        x1 = min(clu_list[i][:,1])
        y1 = min(clu_list[i][:,2])
        x2 = max(clu_list[i][:,1] + clu_list[i][:,3] - 1)
        y2 = max(clu_list[i][:,2] + clu_list[i][:,4] - 1)
        clust_reg[i] = [x1, y1, x2, y2, i]
    return clust_reg










