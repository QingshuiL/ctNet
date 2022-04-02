import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from itertools import cycle


def main(images_path, ann_path):
    anns = os.listdir(ann_path)
    images = os.listdir(images_path)
    
    for i,txt_name in enumerate(anns):
        with open(os.path.join(ann_path,txt_name)) as f:
            lines = f.readlines()
            ann = np.zeros((len(lines), 8), int)
            for j, line in enumerate(lines):
                line = line.strip()
                line = line.split(',')
                ann[j] = line

        # special image
        image_path = os.path.join(images_path,images[i])
        image = cv2.imread(image_path)
        plt.figure(1)
        plt.clf()  # 清楚上面的旧图形
        plt.imshow(image[:,:,::-1])

        for k in range(len(ann)):
            plt.gca().add_patch(plt.Rectangle(xy=(ann[k,0], ann[k,1]),
                width=ann[k,2], #- region_dets[k,0], 
                height=ann[k,3], #- region_dets[k,1],
                edgecolor='r',
                fill=False, linewidth=2))
        plt.show()

if __name__ == "__main__":
    images_path = r'D:\CV\ctNet\data\visdrone2019\cluster_transform'
    anns_path = r'D:\CV\ctNet\data\visdrone2019\cluster_transform_annotations'
    main(images_path, anns_path)