import cv2 as cv 
import numpy as np

np.random.seed(11)


def iou(a, b):
    # get area of a
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

    return iou


# crop and create database
def crop_bbox(img_size, gt, Crop_N=200, L=512, th=0.4):
    # get shape
    W, H = img.shape

    # each crop
    for i in range(Crop_N):
        # get left top x of crop bounding box
        x1 = np.random.randint(W - L)
        # get left top y of crop bounding box
        y1 = np.random.randint(H - L)
        # get right bottom x of crop bounding box
        x2 = x1 + L
        # get right bottom y of crop bounding box
        y2 = y1 + L

        # crop bounding box
        crop_region = np.array((x1, y1, x2, y2))

        # # get IoU between crop box and gt
        # _iou = iou(gt, crop)

        # # assign label
        # if _iou >= th:
        #     cv.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
        #     label = 1
        # else:
        #     cv.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
        #     label = 0

    return crop_region

# read image
img = cv.imread("../xiyi.jpg")
img1 = img.copy()
# gt bounding box
gt = np.array((87, 51, 169, 113), dtype=np.float32)

# get crop bounding box
img = crop_bbox(img, gt, Crop_N=100, L=60, th=0.6)

# draw gt
cv.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,0), 1)
cv.rectangle(img1,(gt[0], gt[1]), (gt[2], gt[3]), (0,255,0), 1)

cv.imshow("result1",img1)
cv.imshow("result", img)
cv.imwrite("out.jpg", img)
cv.waitKey(0)
cv.destroyAllWindows()