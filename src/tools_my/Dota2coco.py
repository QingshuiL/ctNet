import os,glob
import json
import argparse
import PIL.Image
import numpy as np

category_dict = {
    "soccer-ball-field":0, 
    "helicopter":1,
    "swimming-pool":2,
    "roundabout":3,
    "large-vehicle":4,
    "small-vehicle":5,
    "bridge":6,
    "harbor":7,
    "ground-track-field":8,
    "basketball-court":9,
    "tennis-court":10,
    "baseball-diamond":11,
    "storage-tank":12,
    "ship":13,
    "plane":14,
    "container-crane":15
}


attr_dict = dict()
attr_dict["categories"] = [
        {"supercategory": "none", "id": 0, "name": "soccer-ball-field"},
        {"supercategory": "none", "id": 1, "name": "helicopter"},
        {"supercategory": "none", "id": 2, "name": "swimming-pool"},
        {"supercategory": "none", "id": 3, "name": "roundabout"},
        {"supercategory": "none", "id": 4, "name": "large-vehicle"},
        {"supercategory": "none", "id": 5, "name": "small-vehicle"},
        {"supercategory": "none", "id": 6, "name": "bridge"},
        {"supercategory": "none", "id": 7, "name": "harbor"},
        {"supercategory": "none", "id": 8, "name": "ground-track-field"},
        {"supercategory": "none", "id": 9, "name": "basketball-court"},
        {"supercategory": "none", "id": 10, "name": "tennis-court"},
        {"supercategory": "none", "id": 11, "name": "baseball-diamond"},
        {"supercategory": "none", "id": 12, "name": "storage-tank"},
        {"supercategory": "none", "id": 13, "name": "ship"},
        {"supercategory": "none", "id": 14, "name": "plane"},
        {"supercategory": "none", "id": 15, "name": "container-crane"}
    ]


def dota2coco_detection(image_file_list,gt_file_list, fn):
    images = list()
    annotations = list()
    counter = 0
    gt_counter=0
    skip=0
    for i,entry in enumerate(image_file_list):
        print(i)
        img = PIL.Image.open(entry)
        
        if os.stat(gt_file_list[i]).st_size == 0:
            # gt_inform=np.array([1,1,10,10,0,0,0,0])
            skip += 1
            continue
        else:
            gt_inform=np.loadtxt(gt_file_list[i], delimiter=' ', dtype=str, skiprows=2)
        
        # if len(gt_inform) == 1:
        #     continue
        # else:
        #     gt_inform = gt_inform[2:,:]
        
        # gt_inform = gt_inform[2:len(gt_inform)].s
        if len(gt_inform.shape)==1:
            gt_inform=gt_inform.reshape((1,gt_inform.shape[0]))

        counter += 1
        image = dict()
        file_name=os.path.basename(entry)
        # file_name=os.path.split(file_name)[0]
        image['file_name'] = file_name
        image['height'] = img.size[1]
        image['width'] = img.size[0]
        image['id'] = counter
        # image['ori_points'] = [int(gt_inform[0,[6]]), int(gt_inform[0,[7]])]

        bboxs = gt_inform[:,0:8]
        bboxs = bboxs.astype(np.float32)
        category = gt_inform[:,8]
        for gt_i in range(gt_inform.shape[0]):
            gt_counter += 1
            annotation = dict()
            x1 = int(min(bboxs[gt_i, [0,2,4,6]].astype(np.int32)))
            y1 = int(min(bboxs[gt_i, [1,3,5,7]].astype(np.int32)))
            x2 = int(max(bboxs[gt_i, [0,2,4,6]].astype(np.int32)))
            y2 = int(max(bboxs[gt_i, [1,3,5,7]].astype(np.int32)))
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            score = 1
            cat=category_dict[category[gt_i]]

            # if score==0 or cat==0 or cat==11:
            #     annotation["iscrowd"] = 1
            # else:
            #     annotation["iscrowd"] = 0
            
            annotation["iscrowd"] = 0

            annotation["image_id"] = i+1
            annotation['bbox'] = [x1, y1, w, h]
            annotation['area'] = float((x2 - x1) * (y2 - y1))
            annotation['category_id'] = int(cat)
            annotation['ignore'] = annotation["iscrowd"]
            annotation['id'] =gt_counter
            # annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            annotation['segmentation'] = []
            annotations.append(annotation)

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict,cls=NpEncoder)
    with open(fn, "w") as file:
        file.write(json_string)
    print('skip:',skip)


# 转换np格式的数据
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def vis2coco(gt_file_path, image_file_path, out_fn):
    # create visdrone annotation file in COCO format
    print('Loading ground truth file...')
    gt_file_list=glob.glob(gt_file_path)
    gt_file_list.sort()

    print('Loading image file...')
    image_file_list = glob.glob(image_file_path)
    image_file_list.sort()

    dota2coco_detection(image_file_list,gt_file_list, out_fn)

def init_path(task):
    if task == 'val':
        gt_file_path = r"D:\DataSets\Dota数据集\val\labelTxt-v1.5\DOTA-v1.5_val_hbb" + '/*.txt'
        image_file_path = r"D:\DataSets\Dota数据集\val\images" + '/*.png'
        out_fn =  r"D:\DataSets\Dota数据集\val" + '/dota_val.json'
    elif task == 'train':
        gt_file_path = r"D:\DataSets\Dota数据集\train\labelTxt-v1.5\DOTA-v1.5_train" + '/*.txt'
        image_file_path = r"D:\DataSets\Dota数据集\train\images" + '/*.png'
        out_fn =  r"D:\DataSets\Dota数据集\train" + '/dota_train.json'


    return gt_file_path, image_file_path, out_fn
if __name__ == '__main__':
    # train  test  val  cluster_transform   all_train
    # cluster_train_transform 
    # cluster_train_transform_1024_728 
    # cluster_train_transform_1024_824 
    # cluster_transform_val_1024_824

    task = 'train'
    gt_file_path, image_file_path, out_fn = init_path(task)
    vis2coco(gt_file_path, image_file_path, out_fn)

