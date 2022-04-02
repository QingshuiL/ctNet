from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import cv2
from tqdm import tqdm


def black_imags(root_path, phase):
    print('==> initializing visdrone2019 {} data.'.format(phase))
    # 注释文件路径
    annot_path = os.path.join(root_path,'annotations',phase + '_ignore' + '.json')
    vis_Gt = COCO(annot_path)
    imgs = vis_Gt.imgs
    all_anns = vis_Gt.imgToAnns
    imgid_list = vis_Gt.getImgIds()
    num_samples = len(imgs)
    print('Loaded {} {} samples'.format(phase, num_samples))
    # 图片读取路径， 由注释得到
    image_dir = os.path.join(root_path,'images')
    for i in tqdm(imgid_list):
        img_name = imgs[i]['file_name']
        img_path = os.path.join(image_dir,img_name)

        # 保存文件路径
        save_image_path = os.path.join(root_path, 'images_black', img_name)
        im_anns = all_anns[i]
        anns = np.vstack([[ann['bbox']] for ann in im_anns])
        cat_id = np.vstack([ann['category_id'] for ann in im_anns])
        anns = np.concatenate((anns,cat_id),axis=1)
        # 注意ignore类别为 0
        mean_anns = anns[np.where((anns[:,4] == 0))] #| (anns[:,4] == 12))]
        image = cv2.imread(img_path)
        # print('deal iamge:',img_name)
        if image is None:
            print(img_name,' no image read')
            break
        r_mean = int(np.mean(image[:,:,0]))
        g_mean = int(np.mean(image[:,:,1]))
        b_mean = int(np.mean(image[:,:,2]))
        for bbox in mean_anns:
            bbox = bbox[:4].reshape(4,1)
            bbox[2] = bbox[0] + bbox[2] -1
            bbox[3] = bbox[1] + bbox[3] -1
            area = np.array([ [bbox[0],bbox[1]], [bbox[2],bbox[1]], [bbox[2],bbox[3]], [bbox[0],bbox[3]] ])
            cv2.fillPoly(image,[area],(r_mean,g_mean,b_mean))

        show = 0
        if show:
            cv2.imshow(img_name,image)
            cv2.waitKey(0)
        
        cv2.imwrite(save_image_path, image)




if  __name__ == "__main__":

    phase = 'test'
    # 根目录
    root_path = r"D:\WangYi\cv\ctNet\data\visdrone2019"
    black_imags(root_path, phase)
