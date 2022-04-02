import numpy as np
import cv2
import random
import os
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

def CutMix(data_dir):
    for i in tqdm(range(300)):
        CutMix_img_dir = os.path.join(data_dir, 'images_crop_mincat512')
        anns_dir = os.path.join(data_dir, 'annos_crop_mincat512')
        save_img_dir = os.path.join(data_dir, 'Mix1024')
        save_anno_dir = os.path.join(data_dir, 'Mix1024_annos')

        img_name_list = os.listdir(CutMix_img_dir)

        select_img_name = random.sample(img_name_list, 4)

        img_list = []
        for name in select_img_name:
            im_path = os.path.join(CutMix_img_dir, name)
            im = Image.open(im_path).convert("RGB")
            img_list.append(im)
        new_img = Image.new('RGB', (1024, 1024))
        new_img.paste(img_list[0], (0, 0, img_list[0].width, img_list[0].height))
        new_img.paste(img_list[1], (512, 0, img_list[1].width + 512, img_list[1].height))
        new_img.paste(img_list[2], (0, img_list[2].width, 512, img_list[2].height+512))
        new_img.paste(img_list[3], (512, 512, img_list[3].width+512, img_list[3].height+512))
        new_img.save(save_img_dir + '\Mix_{}.jpg'.format(i+1), quality=95,)
        # plt.figure(1)
        # plt.clf()  # 清楚上面的旧图形
        # plt.imshow(new_img)
        # plt.show()

        ann_list = []
        for id, name in enumerate(select_img_name):
            anns_path = os.path.join(anns_dir, name[:-4]+'.txt' )
            anns = np.loadtxt(anns_path, dtype=np.float32, delimiter=',')
            anns = anns.reshape(-1, 8)
            # xyxy -> xywh
            anns[:,[2,3]] = anns[:,[2,3]] - anns[:,[0,1]] +1
            if id == 0:
                pass
            elif id == 1:
                anns[:,0] = anns[:,0] + 512
            elif id == 2:
                anns[:,1] = anns[:,1] + 512
            elif id == 3:
                anns[:,0] = anns[:,0] + 512
                anns[:,1] = anns[:,1] + 512
            ann_list.append(anns)

        all_anns = np.vstack((ann_list))
        np.savetxt(save_anno_dir + '\Mix_{}.txt'.format(i+1), all_anns, fmt='%d', delimiter=',',)

        show=0
        if show:
       
            plt.figure(2)
            plt.clf()  # 清楚上面的旧图形
            plt.imshow(new_img)
            for k in range(len(all_anns)):
                plt.gca().add_patch(plt.Rectangle(xy=(all_anns[k,0], all_anns[k,1]),
                                    width=all_anns[k,2], #- region_dets[k,0], 
                                    height=all_anns[k,3], #- region_dets[k,1],
                                    edgecolor='r',
                                    fill=False, linewidth=2))
            plt.show()

        
if __name__ == "__main__":
    data_dir = r"D:\WangYi\cv\ctNet\data\visdrone2019"
    CutMix(data_dir)
