import torch
import torchvision
import copy
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import os
import warnings

from torchvision.transforms import functional as F
from PIL import Image
from math import inf

from cluster.transform.transform_scale import MonotonicityScaleMatch, ScaleMatch, GaussianScaleMatch, GaussTransfrom
from cluster.transform.BoxList import BoxList


PIL_RESIZE_MODE = {'bilinear': Image.BILINEAR, 'nearest': Image.NEAREST}
min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, filter_ignore=True
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        print("valid image count in dataset: {}".format(len(self.ids)))

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

        self.filter_ignore = filter_ignore  # add by hui

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        # ########################## add by hui ########################################
        img_info = self.get_img_info(idx)
        if 'corner' in img_info:
            img = img.crop(img_info['corner'])
        ################################################################################

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        # ######################### add by hui ####################################
        if self.filter_ignore and anno and "ignore" in anno[0]:  # filter ignore out
            anno = [obj for obj in anno if not obj["ignore"]]
        ###########################################################################

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # masks = [obj["segmentation"] for obj in anno]
        # masks = SegmentationMask(masks, img.size)
        # masks = []
        # target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, img_info

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def fill_img(img):
    w, h = img.size[0], img.size[1]
    w_fill = max(412 - w, 0)
    w_fill = w_fill + 100 if w_fill>0 else 0
    h_fill = max(412 - h, 0)
    h_fill = h_fill + 100 if h_fill>0 else 0
    if w_fill==0 and h_fill ==0:
        img_pasted = img
    else:
        blank_img = Image.new('RGB', (w + w_fill, h + h_fill),(75,75,75)) #数据集图片三通道均值
        blank_img.paste(img, (w_fill//2, h_fill//2, w_fill//2 + w, h_fill//2 + h))
        img_pasted = blank_img
    return img_pasted, w_fill, h_fill

def cutfill_512(img, target):
    imgs = []
    bboxss = []
    w_fills = []
    h_fills = []
    w, h = img.size
    # 不用裁剪可能填充
    if w<=612 and h<=612:
        if w>=412 and h>=412:
            pass
        else:
            img, w_fill, h_fill = fill_img(img)
            bboxs = target.bbox.numpy().astype(np.float32)
            labels = target.extra_fields['labels'].numpy().astype(np.float32)
            bboxs[:,[0,2]] = bboxs[:,[0,2]] + w_fill/2
            bboxs[:,[1,3]] = bboxs[:,[1,3]] + h_fill/2
            print(len(bboxs))
            bboxs = np.concatenate((bboxs, np.ones((len(bboxs),1)), labels.reshape(len(labels),1),
                                    np.zeros((len(bboxs),1)), np.zeros((len(bboxs),1))), axis=1)
            imgs.append(img)
            bboxss.append(bboxs)
            w_fills.append(w_fill)
            h_fills.append(h_fill)
        
        # 裁剪两块再填充
    elif (w<=612 or h<=612) and (w>612 or h>612):
        if w > 612:
            # 分割为512
            img1 = img.crop((0, 0, w/2, h))
            img2 = img.crop((w/2, 0, w, h))
            img1,w_fill1,h_fill1 = fill_img(img1)
            img2,w_fill2,h_fill2 = fill_img(img2)
            
            # 分离 校正 bbox
            bboxs = target.bbox.numpy().astype(np.float32)
            labels = target.extra_fields['labels'].numpy().astype(np.float32)
            centers = np.zeros((len(bboxs), 2))
            centers[:,0] = (bboxs[:,0] + bboxs[:,2]) / 2
            centers[:,1] = (bboxs[:,1] + bboxs[:,3]) / 2

            labels1 = labels[centers[:,0] <= w/2]
            labels2 = labels[centers[:,0] >  w/2]

            bboxs1 = bboxs[centers[:,0] <= w/2]
            bboxs2 = bboxs[centers[:,0] >  w/2]
            bboxs1[:,2] = np.minimum(bboxs1[:,2],w/2)
            bboxs2[:,0] = np.maximum(bboxs2[:,0],w/2)
            bboxs2[:,[0,2]] = bboxs2[:,[0,2]] - w/2
            bboxs1[:,[0,2]] = bboxs1[:,[0,2]] + w_fill1//2
            bboxs1[:,[1,3]] = bboxs1[:,[1,3]] + h_fill1//2
            bboxs2[:,[0,2]] = bboxs2[:,[0,2]] + w_fill2//2
            bboxs2[:,[1,3]] = bboxs2[:,[1,3]] + h_fill2//2
            
            bboxs1 = np.concatenate((bboxs1, np.ones((len(bboxs1),1)), labels1.reshape(len(labels1),1), 
                                    np.zeros((len(bboxs1),1)), np.zeros((len(bboxs1),1))), axis=1)
            bboxs2 = np.concatenate((bboxs2, np.ones((len(bboxs2),1)), labels2.reshape(len(labels2),1),
                                    np.zeros((len(bboxs2),1)), np.zeros((len(bboxs2),1))), axis=1)
            imgs.extend([img1, img2])
            bboxss.extend([bboxs1,bboxs2])
            w_fills.extend([w_fill1,w_fill2])
            h_fills.extend([h_fill1,h_fill2])

        elif h > 612:
            img1 = img.crop((0, 0, w, h/2))
            img2 = img.crop((0, h/2, w, h))
            img1,w_fill1,h_fill1 = fill_img(img1)
            img2,w_fill2,h_fill2 = fill_img(img2)
            
            bboxs = target.bbox.numpy().astype(np.float32)
            labels = target.extra_fields['labels'].numpy().astype(np.float32)
            centers = np.zeros((len(bboxs), 2))
            centers[:,0] = (bboxs[:,0] + bboxs[:,2]) / 2
            centers[:,1] = (bboxs[:,1] + bboxs[:,3]) / 2
            labels1 = labels[centers[:,1] <= h/2]
            labels2 = labels[centers[:,1] >  h/2]

            bboxs1 = bboxs[centers[:,1] <= h/2]
            bboxs2 = bboxs[centers[:,1] >  h/2]
            bboxs1[:,3] = np.minimum(bboxs1[:,3], h/2)
            bboxs2[:,1] = np.maximum(bboxs2[:,1], h/2)

            bboxs2[:,[1,3]] = bboxs2[:,[1,3]] - h/2

            bboxs1[:,[0,2]] = bboxs1[:,[0,2]] + w_fill1//2
            bboxs1[:,[1,3]] = bboxs1[:,[1,3]] + h_fill1//2
            bboxs2[:,[0,2]] = bboxs2[:,[0,2]] + w_fill2//2
            bboxs2[:,[1,3]] = bboxs2[:,[1,3]] + h_fill2//2

            bboxs1 = np.concatenate((bboxs1, np.ones((len(bboxs1),1)), labels1.reshape(len(labels1),1), 
                                    np.zeros((len(bboxs1),1)), np.zeros((len(bboxs1),1))), axis=1)
            bboxs2 = np.concatenate((bboxs2, np.ones((len(bboxs2),1)), labels2.reshape(len(labels2),1),
                                    np.zeros((len(bboxs2),1)), np.zeros((len(bboxs2),1))), axis=1)

            imgs.extend([img1, img2])
            bboxss.extend([bboxs1,bboxs2])
            w_fills.extend([w_fill1,w_fill2])
            h_fills.extend([h_fill1,h_fill2])
        # 裁剪四块再填充
    elif w>=612 and h>=612:
        img1 = img.crop((0, 0, w/2, h/2))
        img2 = img.crop((w/2, 0, w, h/2))
        img3 = img.crop((0, h/2, w/2, h))
        img4 = img.crop((w/2, h/2, w, h))

        img1,w_fill1,h_fill1 = fill_img(img1)
        img2,w_fill2,h_fill2 = fill_img(img2)
        img3,w_fill3,h_fill3 = fill_img(img3)
        img4,w_fill4,h_fill4 = fill_img(img4)

        bboxs = target.bbox.numpy().astype(np.float32)
        labels = target.extra_fields['labels'].numpy().astype(np.float32)
        centers = np.zeros((len(bboxs), 2))
        centers[:,0] = (bboxs[:,0] + bboxs[:,2]) / 2
        centers[:,1] = (bboxs[:,1] + bboxs[:,3]) / 2
        labels1 = labels[np.where((centers[:,0] <= w/2) & (centers[:,1] <= h/2))]
        labels2 = labels[np.where((centers[:,0] >  w/2) & (centers[:,1] <= h/2))]
        labels3 = labels[np.where((centers[:,0] <= w/2) & (centers[:,1] >  h/2))]
        labels4 = labels[np.where((centers[:,0] >  w/2) & (centers[:,1] >  h/2))]

        bboxs1 = bboxs[np.where((centers[:,0] <= w/2) & (centers[:,1] <= h/2))]
        bboxs2 = bboxs[np.where((centers[:,0] >  w/2) & (centers[:,1] <= h/2))]
        bboxs3 = bboxs[np.where((centers[:,0] <= w/2) & (centers[:,1] >  h/2))]
        bboxs4 = bboxs[np.where((centers[:,0] >  w/2) & (centers[:,1] >  h/2))]

        bboxs1[:,2] = np.minimum(bboxs1[:,2], w/2)
        bboxs1[:,3] = np.minimum(bboxs1[:,3], h/2)
        bboxs1[:,[0,2]] = bboxs1[:,[0,2]] + w_fill1//2
        bboxs1[:,[1,3]] = bboxs1[:,[1,3]] + h_fill1//2

        bboxs2[:,0] = np.maximum(bboxs2[:,0], w/2)
        bboxs2[:,3] = np.minimum(bboxs2[:,3], h/2)
        bboxs2[:,[0,2]] = bboxs2[:,[0,2]] - w/2
        bboxs2[:,[0,2]] = bboxs2[:,[0,2]] + w_fill2//2
        bboxs2[:,[1,3]] = bboxs2[:,[1,3]] + h_fill2//2

        bboxs3[:,1] = np.maximum(bboxs3[:,1], h/2)
        bboxs3[:,2] = np.minimum(bboxs3[:,2], w/2)
        bboxs3[:,[1,3]] = bboxs3[:,[1,3]] - h/2
        bboxs3[:,[0,2]] = bboxs3[:,[0,2]] + w_fill3//2
        bboxs3[:,[1,3]] = bboxs3[:,[1,3]] + h_fill3//2

        bboxs4[:,0] = np.maximum(bboxs4[:,0], w/2)
        bboxs4[:,1] = np.maximum(bboxs4[:,1], h/2)
        bboxs4[:,[0,2]] = bboxs4[:,[0,2]] - w/2
        bboxs4[:,[1,3]] = bboxs4[:,[1,3]] - h/2
        bboxs4[:,[0,2]] = bboxs4[:,[0,2]] + w_fill1//2
        bboxs4[:,[1,3]] = bboxs4[:,[1,3]] + h_fill1//2

        bboxs1 = np.concatenate((bboxs1, np.ones((len(bboxs1),1)), labels1.reshape(len(labels1),1), 
                                    np.zeros((len(bboxs1),1)), np.zeros((len(bboxs1),1))), axis=1)
        bboxs2 = np.concatenate((bboxs2, np.ones((len(bboxs2),1)), labels2.reshape(len(labels2),1),
                                    np.zeros((len(bboxs2),1)), np.zeros((len(bboxs2),1))), axis=1)
        bboxs3 = np.concatenate((bboxs3, np.ones((len(bboxs3),1)), labels3.reshape(len(labels3),1),
                                    np.zeros((len(bboxs3),1)), np.zeros((len(bboxs3),1))), axis=1)
        bboxs4 = np.concatenate((bboxs4, np.ones((len(bboxs4),1)), labels4.reshape(len(labels4),1),
                                    np.zeros((len(bboxs4),1)), np.zeros((len(bboxs4),1))), axis=1)

        imgs.extend([img1,img2,img3,img4])
        bboxss.extend([bboxs1,bboxs2,bboxs3,bboxs4])
        w_fills.extend([w_fill1,w_fill2,w_fill3,w_fill4])
        h_fills.extend([h_fill1,h_fill2,h_fill3,h_fill4])

    return imgs, bboxss, w_fills, h_fills

if __name__ == "__main__":

    # 主要跟注释文件有关跟image无关
    data_dir = r'C:\Users\10295\CV\ctNet\data'
    vis_dir = os.path.join(data_dir,'visdrone2019')
    coco_dir = os.path.join(data_dir,'coco2017')
    visClu_anno_file = os.path.join(vis_dir,'annotations','cluster_train.json')
    coco_anno_file = os.path.join(coco_dir,'annotations','instances_train2017.json')
    save_im_dir = os.path.join(vis_dir,'cluster_transform')
    save_ann_dir = os.path.join(vis_dir,'cluster_transform_annotations')

    # 按照dst分布转换src分布
    # sm = MonotonicityScaleMatch(src_anno_file, dst_anno_file)
    transform = GaussTransfrom(mu=79.5, sigma=5, scale_range=(0.5, 3.))
    # transform = Compose([sm])
    dataset = COCODataset(visClu_anno_file, vis_dir + '\\images', True)
    for ind, (img, target, img_info) in  enumerate(dataset):
        # if ind<2:
        #     continue
        print('{}/{}'.format(ind,len(dataset)))
        ori_size = np.array(img.size)
        
        show_1 = 0
        if show_1:
            plt.figure(10)
            plt.imshow(img)
            plt.title(img_info['file_name'] + '_ori')
            show_bbox = target.bbox.numpy().astype(np.float32)
            for k in range(len(show_bbox)):
                plt.gca().add_patch(plt.Rectangle(xy=(show_bbox[k,0], show_bbox[k,1]),
                                    width=show_bbox[k,2] - show_bbox[k,0], 
                                    height=show_bbox[k,3] - show_bbox[k,1],
                                    edgecolor='r',
                                    fill=False, linewidth=2))
            plt.show()

        img, target, scale = transform(img, target)

        if len(target.bbox) == 0:
            print('注释为空跳过该图片')
            continue

        show_2 = 0
        if show_2:
            plt.figure(11)
            plt.imshow(img)
            plt.title(img_info['file_name'] + '_tranf')
            show_bbox = target.bbox.numpy().astype(np.float32)
            for k in range(len(show_bbox)):
                plt.gca().add_patch(plt.Rectangle(xy=(show_bbox[k,0], show_bbox[k,1]),
                                    width=show_bbox[k,2] - show_bbox[k,0], 
                                    height=show_bbox[k,3] - show_bbox[k,1],
                                    edgecolor='r',
                                    fill=False, linewidth=2))
            plt.show()

        imgs, bboxss, w_fills, h_fills = cutfill_512(img, target)

        for i, save_im in enumerate(imgs):
            save_im_name = img_info['file_name'][:-4] + '_{}'.format(i) + '.jpg'
            save_ann_name = img_info['file_name'][:-4] + '_{}'.format(i) + '.txt'
            # bbox 为 xyxyx 转为 xywh
            Bboxs = bboxss[i]
            Bboxs[:,[2,3]] = Bboxs[:,[2,3]] - Bboxs[:,[0,1]] + 1

            if len(Bboxs)==0:
                continue
            np.savetxt(os.path.join(save_ann_dir,save_ann_name),Bboxs,fmt="%d",delimiter=",")
            save_im.save(os.path.join(save_im_dir,save_im_name), quality=95)
            
        show = 0
        if show:
            plt.figure(0)
            plt.imshow(img)
            plt.title('img')
            for j, im in enumerate(imgs):
                show_bbox = bboxss[j]
                plt.figure(j+1)  
                plt.imshow(im)
                for k in range(len(show_bbox)):
                    plt.gca().add_patch(plt.Rectangle(xy=(show_bbox[k,0], show_bbox[k,1]),
                                    width=show_bbox[k,2], #- show_bbox[k,0], 
                                    height=show_bbox[k,3], #- show_bbox[k,1],
                                    edgecolor='r',
                                    fill=False, linewidth=2))
            plt.show()
 