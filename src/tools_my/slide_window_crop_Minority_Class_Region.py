import os
import pandas as pd
from PIL import Image
import numpy as np


images_dir = r'D:\WangYi\cv\ctNet\data\visdrone2019\images'
annotations_dir = r'D:\WangYi\cv\ctNet\data\visdrone2019\anns_train_test'
annos = os.listdir(annotations_dir)
images = os.listdir(images_dir)
im = []
for i,txt_name in enumerate(annos):
    txt_name = txt_name.split('.')
    im.append( txt_name[0])

def stat_nums(s1,t1,s2,t2,anno,min_class):
    anno = anno[anno[:, 5] == min_class]
    a = 0
    for i in range(len(anno)):
        if s1<= anno[i,6]<=s2 and t1<= anno[i,7]<=t2:
            a += 1 
    return a


def slide_window_crop_Minority_Class_Region(image,anno,interval,sw_size, min_class):
    th_rate = 0.2
    th_nums = 20
    if min_class == 3:
        th_rate = 0.15
        th_nums = 15
    elif min_class == 8:
        th_rate = 0.1
        th_nums = 10

    w,h = image.size[0],image.size[1]
    s1,t1,s2,t2 = 0,0,512,512
    interval = interval
    a = 0
    R_x0 = 0
    R_y0 = 0
    R_x1 = 0
    R_y1 = 0
    while True:
        if s2<w:
            a1 = stat_nums(s1,t1,s2,t2,anno,min_class)
            if a1 > a :
                a = a1
                R_x0 = s1
                R_y0 = t1
                R_x1 = s2
                R_y1 = t2 
            
            s1 = s1 + interval
            s2 = s2 + interval
        else:
            if t2<h:
                s1 = 0
                s2 = 512
                t1 += interval
                t2 += interval
            else:
                if (a > th_rate * len(anno)) or (a > th_nums):    #认定是可以作为裁剪区域的标准
                    #return image.crop([R_x0,R_y0,R_x1,R_y1]),crop_anno(R_x0,R_y0,R_x1,R_y1,anno)
                    return a,R_x0,R_y0,R_x1,R_y1
                else:
                    return None


#问题1：是以IOU以判断一个目标是否属于这个区域还是使用中心点在区域里面则属于这个区域
#修改原始的注释文件得到裁剪图片的注释文件
def crop_anno(R_x0,R_y0,R_x1,R_y1,anno):
    anno[:,0] = anno[:,0]-R_x0
    anno[:,2] = anno[:,2]-R_x0
    anno[:,6] = anno[:,6]-R_x0
    anno[:,1] = anno[:,1]-R_y0
    anno[:,3] = anno[:,3]-R_y0
    anno[:,7] = anno[:,7]-R_y0
    anno = anno[np.where(  (0<=anno[:,6]) & (anno[:,6]<=512) & (0<=anno[:,7]) & (anno[:,7]<=512))]
    anno[anno[:,0]<0,0] = 0
    anno[anno[:,1]<0,1] = 0
    anno[anno[:,2]>512,2] = 512
    anno[anno[:,3]>512,3] = 512
    return anno 

min_cat = [2, 3, 8]
for cat in min_cat:
    for i, t_name in enumerate(im):
        # # 2 3 8
        # min_cat = 2
        img_name = os.path.join(images_dir, '{}.jpg'.format(t_name))
        txt_name = os.path.join(annotations_dir, '{}.txt'.format(t_name))
        images_dir_c = r'D:\WangYi\cv\ctNet\data\visdrone2019\images_crop_mincat512'
        annotations_dir_c = r'D:\WangYi\cv\ctNet\data\visdrone2019\annos_crop_mincat512'
        # .png
        # save_image = os.path.join(images_dir, '{}.jpg'.format(t_name))
        img_path = os.path.join(images_dir_c, 
            t_name + '_cat{}'.format(cat) + ".png")
        save_txt = os.path.join(annotations_dir_c, 
            t_name + '_cat{}'.format(cat) + ".txt")
        # read image
        image = Image.open(img_name).convert("RGB")
        # read annotation
        annotation = pd.read_csv(txt_name, header=None)
        annotation = np.array(annotation)[:, :8]
        annotation = annotation[annotation[:, 5] != 11]   #剔除忽略区域
        annotation[:,2] = annotation[:,0] + annotation[:,2]   #转换为左上角和右下角坐标的形式
        annotation[:,3] = annotation[:,1] + annotation[:,3]
        annotation[:,6] = (annotation[:,0] + annotation[:,2])/2.0   #计算目标的中心点
        annotation[:,7] = (annotation[:,1] + annotation[:,3])/2.0
        #annotation = np.column_stack((annotation,annotation_8))   #在最后添加中心点这一列
        #annotation = np.column_stack((annotation,annotation_9))
        coodir = slide_window_crop_Minority_Class_Region(image,annotation, 32, 512, cat)  
        if coodir is not None:
            print(i)
            img_c = image.crop([coodir[1],coodir[2],coodir[3],coodir[4]])
            anno_c = crop_anno(coodir[1],coodir[2],coodir[3],coodir[4],annotation)
            img_c.save(img_path)
            np.savetxt(save_txt,anno_c,fmt='%d',delimiter=',',)