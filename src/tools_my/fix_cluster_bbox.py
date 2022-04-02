import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

def get_annotation(phase):

    root = "F:/BaiduNetdiskDownload/visDrone2019"
    txt_dir = os.path.join(root,'VisDrone2019-DET-' + phase,'cluster_annotations')
    txt_list = glob.glob(txt_dir + "/*.txt")
    txt_list = np.sort(txt_list)
    for index, line in enumerate(txt_list):
        print(line)
        dets = np.loadtxt(line, dtype=np.float32, delimiter=',')
        if len(dets) > 1:
            dets[:,2] = dets[:,2] - dets[:,0]
            dets[:,3] = dets[:,3] - dets[:,1]
        elif len(dets) == 1:
            dets[2] = dets[2] - dets[0]
            dets[3] = dets[3] - dets[1]

        np.savetxt(line, dets, fmt='%d',delimiter=',')

if __name__ == "__main__":
    get_annotation('train')