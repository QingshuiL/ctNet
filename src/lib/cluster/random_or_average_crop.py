

import numpy as np 
import torch



def max_iou(regions, b):
    if len(regions)==0:
        return 0
    # get area of a
    maxiou = 0
    for a in regions:
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        # get area of b
        area_b = (b[2] - b[0]) * (b[3] - b[1])

        # get left top x of IoU
        iou_x1 = np.maximum(a[0], b[0])
        # get left top y of IoU
        iou_y1 = np.maximum(a[1], b[1])
        # get right bottom of IoU
        iou_x2 = np.minimum(a[2], b[2])
        # get right bottom of IoU
        iou_y2 = np.minimum(a[3], b[3])

        # get width of IoU
        iou_w = iou_x2 - iou_x1
        # get height of IoU
        iou_h = iou_y2 - iou_y1

        # get area of IoU
        area_iou = iou_w * iou_h
        # get overlap ratio between IoU and all area
        iou = area_iou / (area_a + area_b - area_iou)
        if iou>maxiou:
            maxiou = iou
    return maxiou


def average_crop(img_size):
    w = img_size[0]
    h = img_size[1]
    crop_coord1 = np.array((0, 0, w/2, h/2)).reshape(-1,4)
    crop_coord2 = np.array((w/2, 0, w, h/2)).reshape(-1,4)
    crop_coord3 = np.array((0, h/2, w/2, h)).reshape(-1,4)
    crop_coord4 = np.array((w/2, h/2, w, h)).reshape(-1,4)
    crop_regions = [crop_coord1, crop_coord2, crop_coord3, crop_coord4]
    return crop_regions, 1


def random_crop(img_size, th=0.5):
    W,H = img_size
    crop_w = W/2
    crop_h = H/2
    Crop_N = 4
    crop_regions = []
    num_regions = 0
    while num_regions<Crop_N:
        x1 = np.random.randint(W - crop_w)
        # get left top y of crop bounding box
        y1 = np.random.randint(H - crop_h)
            # get right bottom x of crop bounding box
        x2 = x1 + crop_w
            # get right bottom y of crop bounding box
        y2 = y1 + crop_w
        crop_region = np.array((x1, y1, x2, y2)).reshape(-1,4)
        maxiou = max_iou(crop_regions,crop_region)
        if maxiou <= th:
            crop_regions.append(crop_region)
            num_regions += 1
    return crop_regions, 1

