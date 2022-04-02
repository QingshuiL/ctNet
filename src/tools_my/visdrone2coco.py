import os,glob
import json
import argparse
import PIL.Image
import numpy as np


attr_dict = dict()
attr_dict["categories"] = [
        {"supercategory": "none", "id": 0, "name": "ignored regions"},
        {"supercategory": "none", "id": 1, "name": "pedestrian"},
        {"supercategory": "none", "id": 2, "name": "people"},
        {"supercategory": "none", "id": 3, "name": "bicycle"},
        {"supercategory": "none", "id": 4, "name": "car"},
        {"supercategory": "none", "id": 5, "name": "van"},
        {"supercategory": "none", "id": 6, "name": "truck"},
        {"supercategory": "none", "id": 7, "name": "tricycle"},
        {"supercategory": "none", "id": 8, "name": "awning - tricycle"},
        {"supercategory": "none", "id": 9, "name": "bus"},
        {"supercategory": "none", "id": 10, "name": "motor"},
        {"supercategory": "none", "id": 11, "name": "others"}
    ]


def visdrone2coco_detection(image_file_list,gt_file_list, fn):
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
            gt_inform=np.loadtxt(gt_file_list[i],delimiter=',')

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
        image['ori_points'] = [int(gt_inform[0,[6]]), int(gt_inform[0,[7]])]

        for gt_i in range(gt_inform.shape[0]):
            gt_counter +=1
            annotation = dict()
            x1 = int(gt_inform[gt_i,0])
            y1 = int(gt_inform[gt_i, 1])
            w = int(gt_inform[gt_i, 2])
            h = int(gt_inform[gt_i, 3])
            x2= int(x1+w-1)
            y2= int(y1+h-1)
            score=gt_inform[gt_i, 4]
            cat=gt_inform[gt_i, 5]

            if score==0 or cat==0 or cat==11:
                annotation["iscrowd"] = 1
            else:
                annotation["iscrowd"] = 0
            
            # 临时修改 注意改回
            # annotation["iscrowd"] = 0

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

    visdrone2coco_detection(image_file_list,gt_file_list, out_fn)

def init_path(task):
    if task == 'train':
        gt_file_path = r"D:\WangYi\Datasets\VisDrone2018-DET-train\annotations" + '/*.txt'
        image_file_path = r"D:\WangYi\Datasets\VisDrone2018-DET-train\images" + '/*.jpg'
        out_fn =  r"D:\WangYi\Datasets\VisDrone2018-DET-train\json" + '/train.json'
    elif task == 'test':
        gt_file_path = r"D:\WangYi\Datasets\VisDrone2018-DET-test\annotations" + '/*.txt'
        image_file_path = r"D:\WangYi\Datasets\VisDrone2018-DET-test\images" + '/*.jpg'
        out_fn =  r"D:\WangYi\Datasets\VisDrone2018-DET-test\json" + '/test.json'
    elif task == 'all_train':
        gt_file_path = r"D:\WangYi\cv\ctNet\data\visdrone2019\anns_train_test" + '/*.txt'
        image_file_path = r"D:\WangYi\cv\ctNet\data\visdrone2019\images_black" + '/*.jpg'
        out_fn =  r"D:\WangYi\cv\ctNet\data\visdrone2019\annotations" + '/all_train.json'
    elif task == 'val':
        gt_file_path = r"D:\WangYi\Datasets\VisDrone2018-DET-val\annotations" + '/*.txt'
        image_file_path = r"D:\WangYi\Datasets\VisDrone2018-DET-val\images" + '/*.jpg'
        out_fn =  r"D:\WangYi\Datasets\VisDrone2018-DET-val\json" + '/val.json'
    elif task == 'cluster_transform':
        gt_file_path = r"D:\CV\ctNet\data\visdrone2019\cluster_transform_annotations" + '/*.txt'
        image_file_path = r"D:\CV\ctNet\data\visdrone2019\cluster_transform" + '/*.jpg'
        out_fn =  r"D:\CV\ctNet\data\visdrone2019\annotations" + '/cluster_transform.json'
    elif task == 'cluster_val':
        gt_file_path = r"H:\Datasets\VisDrone2019\VisDrone2019-DET-val\cluster_val_annotations" + '/*.txt'
        image_file_path = r"H:\Datasets\VisDrone2019\VisDrone2019-DET-val\cluster_val" + '/*.jpg'
        out_fn =  r"H:\Datasets\VisDrone2019\VisDrone2019-DET-val\json" + '/cluster_val.json'
        # 需要对聚类区域进行尺度变换以后再做 vis2coco
    elif task == 'cluster_train_transform':
        gt_file_path = r"C:\Users\10295\CV\ctNet\data\visdrone2019\cluster_transform_annotations" + '/*.txt'
        image_file_path = r"C:\Users\10295\CV\ctNet\data\visdrone2019\cluster_transform" + '/*.jpg'
        out_fn =  r"C:\Users\10295\CV\ctNet\data\visdrone2019\annotations" + '/cluster_train_transform.json'
    elif task == 'cluster_train_transform_1024_724':
        gt_file_path = r"C:\Users\10295\CV\ctNet\data\visdrone2019\cluster_transform_annotations_1024-724" + '/*.txt'
        image_file_path = r"C:\Users\10295\CV\ctNet\data\visdrone2019\cluster_transform_1024-724" + '/*.jpg'
        out_fn =  r"C:\Users\10295\CV\ctNet\data\visdrone2019\annotations" + '/cluster_train_transform_724.json'
    elif task == 'cluster_train_transform_1024_824':
        gt_file_path = r"C:\Users\10295\CV\ctNet\data\visdrone2019\cluster_transform_annotations_1024-824" + '/*.txt'
        image_file_path = r"C:\Users\10295\CV\ctNet\data\visdrone2019\cluster_transform_1024-824" + '/*.jpg'
        out_fn =  r"C:\Users\10295\CV\ctNet\data\visdrone2019\annotations" + '/cluster_train_transform_824.json'
    elif task == 'cluster_transform_val_1024_824':
        gt_file_path = r"C:\Users\10295\CV\ctNet\data\visdrone2019\cluster_transform_val_annotations_1024-824" + '/*.txt'
        image_file_path = r"C:\Users\10295\CV\ctNet\data\visdrone2019\cluster_transform_val_1024-824" + '/*.jpg'
        out_fn =  r"C:\Users\10295\CV\ctNet\data\visdrone2019\annotations" + '/cluster_val_transform_824.json'

    return gt_file_path, image_file_path, out_fn
if __name__ == '__main__':
    # train  test  val  cluster_transform   all_train
    # cluster_train_transform 
    # cluster_train_transform_1024_728 
    # cluster_train_transform_1024_824 
    # cluster_transform_val_1024_824

    task = 'cluster_transform'
    gt_file_path, image_file_path, out_fn = init_path(task)
    vis2coco(gt_file_path, image_file_path, out_fn)

