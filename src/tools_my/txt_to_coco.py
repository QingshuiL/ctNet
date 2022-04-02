import json
import os
import cv2

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，
# classes.txt(类别标签),以及annotations文件夹
# (如果没有则会自动创建，用于保存最后的json)

# 训练集和验证集划分的界线PP
# split = 8000

# 打开类别标签
# with open(os.path.join(root_path, 'classes.txt')) as f:
#     classes = f.read().strip().split()
classes = ['car', 'truck', 'bus']


def main(phase, root_path):
    # 建立类别标签和数字id的对应关系
    dataset = {'categories': [], 'images': [], 'annotations': []}
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i+1, 'name': cls, 'supercategory': ''})

    # 读取images文件夹的图片名称
    image_dir = [f for f in os.listdir(os.path.join(root_path, 'images'))]
    indexes = []
    for i, line in enumerate(image_dir):
        indexes.append(os.path.splitext(line)[0])

    # # 处理train序列
    # train_dir = [f for f in os.listdir(os.path.join(root_path, 'label', 'train'))]
    # train_list = []
    # for i, line in enumerate(train_dir):
    #     train_list.append(os.path.splitext(line)[0])
    # # 处理test序列
    # test_dir = [f for f in os.listdir(os.path.join(root_path, 'label', 'train'))]
    # test_list = []
    # for i, line in enumerate(test_dir):
    #     test_list.append(os.path.splitext(line)[0])
    # # 判断是建立训练集还是验证集
    # if phase == 'train':
    #     indexes = [line for i, line in enumerate(indexes) if line in train_list]
    # elif phase == 'test':
    #     indexes = [line for i, line in enumerate(indexes) if line in test_list]

    # 读取Bbox信息
    with open(os.path.join(root_path, 'annos.txt')) as tr:
        annos = tr.readlines()


    for k, index in enumerate(indexes):

        # 读取bbox信息
        with open(os.path.join(root_path, 'lable', phase, index + 'txt'), 'r') as f:
            annos = f.readlines()
        if annos is None:
            continue

        # 用opencv读取图片，得到图像的宽和高
        im = cv2.imread(os.path.join(root_path, 'images', index+'jpg'))
        height, width, _ = im.shape

        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': index,
                                'id': k,
                                'width': width,
                                'height': height})

    for ii, anno in enumerate(annos):
        parts = anno.strip().split()
        # anns:bbox score cat ig ob 
        # 如果图像的名称和标记的名称对上，则添加标记
        if parts[0] == index:
            # 类别
            cls_id = parts[4]
            # x_min
            x1 = float(parts[0])
            # y_min
            y1 = float(parts[1])
            # # x_max
            # x2 = float(parts[4])
            # # y_max
            # y2 = float(parts[5])

            width = float(parts[2])
            height = float(parts[3])
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls_id),
                'id': i,
                'image_id': k,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': []
            })



    folder = os.path.join(root_path, 'annotations')

    if not os.path.exists(folder):
        os.makedirs(folder)

    json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))

    with open(json_name, 'w') as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    # train val test
    phase = 'train'
    root = r'D:\WangYi\Datasets\UAVDT'
    main(phase, root)