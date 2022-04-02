import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def vis_clust(dets_list, clust_ori_reg, image, num_class):
    clust = clust_ori_reg[0]
    clust = clust.reshape(clust.shape[0],clust.shape[1])
    # dets = dets_list[1].copy()
    dets = dets_list.copy()
    plt.figure(1)
    plt.clf()  # 清楚上面的旧图形
    plt.imshow(image[:, :, ::-1])
    # cycle把一个序列无限重复下去
    colors = cycle('bgrcmyk')

    # 画聚类方框
    for k, color in zip(range(clust.shape[0]), colors):
        plt.gca().add_patch(plt.Rectangle(xy=(clust[k,0], clust[k,1]),
                                  width=clust[k,2] - clust[k,0], 
                                  height=clust[k,3] - clust[k,1],
                                  edgecolor=color,
                                  fill=False, linewidth=2))
        plt.text(clust[k,0], clust[k,1], k,color='r')

    # 画目标检测框
    for i in range(1,11):
        det_cat = dets[i]
        det_cat = det_cat[det_cat[:,4]>0.2]
        for j in range(len(det_cat)):
            plt.gca().add_patch(plt.Rectangle(xy=(det_cat[j,0], det_cat[j,1]),
                                  width=det_cat[j,2] - det_cat[j,0], 
                                  height=det_cat[j,3] - det_cat[j,1],
                                  edgecolor='r',
                                  fill=False, linewidth=2))
    
    plt.title('vision number of clusters: %d' % clust.shape[0])
    plt.show()


def select_clsuter(clust):
    n = len(clust)
    regs = []
    clu_id = 0
    del_num = 0
    for i in range(len(clust)):
        clu_reg = clust[i]
        w = clu_reg[2] - clu_reg[0]
        h = clu_reg[3] - clu_reg[1]
        max_side = max(w,h)
        area = w * h
        wh_scale = w / h
        if max_side < 100 or area < 10000 or wh_scale >= 4 or wh_scale <= 0.25:
            del_num += 1
            continue
        clu_reg[4] = clu_id
        clu_reg = clu_reg.reshape(1,5)
        clu_id += 1
        regs.append(clu_reg)
    if len(regs) == 0:
        clust_reg = None
        clust_num = 0
        return clust_reg, clust_num

    clust_reg = np.concatenate(regs, axis=0).astype(np.int32)
    clust_num = len(clust_reg)
    if del_num != 0:
        print('del the number of cluster is {}'.format(del_num))
    assert (clust_num + del_num) == n , 'the process of select cluster are error'
    return clust_reg, clust_num


def cut_clust_gen_Dets(opt ,phase, clust_ori_reg, clusted_ori_dets, image, image_or_path_or_tensor,vis): 

    phase = phase
    root = r"H:\Datasets\VisDrone2019"
    image_out_path = os.path.join(root,'VisDrone2019-DET-' + phase,'cluster_'+phase)
    save_txt_dir = os.path.join(root,'VisDrone2019-DET-' + phase,'cluster_'+phase+'_annotations')
    alpha_name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l']
    res = os.path.split(image_or_path_or_tensor)
    image_id = res[1][:6]

    clust = clust_ori_reg[0]
    clust = clust.reshape(clust.shape[0],clust.shape[1]) # ragion:x0,y0,x1,y1; cluster
    num_cluster = len(clust)
    clust[:,0] = np.maximum(clust[:,0],0)
    clust[:,1] = np.maximum(clust[:,1],0)
    clust[:,2] = np.maximum(clust[:,2],0)
    clust[:,3] = np.maximum(clust[:,3],0)
    clust, num_cluster = select_clsuter(clust)
    if num_cluster == 0:
        print('select_clsuter is 0')
        return None, None, None

    # 保存裁剪的聚类区域
    x0 = clust[:,0].copy()
    y0 = clust[:,1].copy()
    x1 = clust[:,2].copy()
    y1 = clust[:,3].copy()
    # ori_clu_x0y0 = np.concatenate((x0,y0), axis=1)
    print("start cut_gen_clustDets", num_cluster)
    images_cropped = []
    for i in range(num_cluster):
        cropped = image[y0[i]:y1[i], x0[i]:x1[i]]
        clu_reg_path = os.path.join(image_out_path,image_id + str(alpha_name[i]) + ".jpg")
        images_cropped.append(cropped)
        if opt.cluster == 1:
            cv2.imwrite(clu_reg_path, cropped)
    
    # 处理聚类区域的注释文件
    if opt.cluster == 2:
        # 使用检测出来的bbox
        dets = clusted_ori_dets[0,:,1:]
    else:
        # 使用真实的GT
        dets = get_annotation(phase, image_id, root)
        dets[:,2] = dets[:,0] + dets[:,2]
        dets[:,3] = dets[:,1] + dets[:,3] # xyxy, scorce, catgriage, _, _

    clu_reg_dets = []
    for i in range(num_cluster):
        region = clust[i,:]
        x0 = region[0]
        y0 = region[1]
        x1 = region[2]
        y1 = region[3]
        clust_id = region[4]
        ct_x = (dets[:,2] + dets[:,0]) / 2 
        ct_y = (dets[:,1] + dets[:,3]) / 2 
        region_dets = dets[np.where((ct_x>=x0) & (ct_x<=x1) & (ct_y>=y0) & (ct_y<=y1))]
        region_dets[:,0] = np.maximum(region_dets[:,0] - x0,0)
        region_dets[:,1] = np.maximum(region_dets[:,1] - y0,0)
        region_dets[:,2] = np.maximum(region_dets[:,2] - x0,0)
        region_dets[:,3] = np.maximum(region_dets[:,3] - y0,0)
        region_dets[:,2] = region_dets[:,2] - region_dets[:,0] 
        region_dets[:,3] = region_dets[:,3] - region_dets[:,1]
        region_dets[:,6] = np.full((len(region_dets)), x0)
        region_dets[:,7] = np.full((len(region_dets)), y0) # xyxy,scorce,cat,clusterx0,clustery0
        # region_dets = region_dets.astype(np.int32)
        if opt.cluster == 1:
            clu_txt_path = os.path.join(save_txt_dir,image_id + str(alpha_name[i]) + ".txt")
            np.savetxt(clu_txt_path,region_dets,fmt='%d',delimiter=',',)
        clu_reg_dets.append(region_dets)
        #画图验证
        if vis:
            import matplotlib.pyplot as plt
            from itertools import cycle
            clu_image_path = os.path.join(image_out_path,image_id + str(alpha_name[i]) + ".jpg")
            # image = cv2.imread(clu_image_path)
        
            plt.figure(1)
            plt.clf()  # 清楚上面的旧图形
            plt.imshow(images_cropped[i][:,:,::-1])
            for k in range(len(region_dets)):
                plt.gca().add_patch(plt.Rectangle(xy=(region_dets[k,0], region_dets[k,1]),
                                    width=region_dets[k,2], #- region_dets[k,0], 
                                    height=region_dets[k,3], #- region_dets[k,1],
                                    edgecolor='r',
                                    fill=False, linewidth=2))
            plt.show()

    if opt.cluster == 2:
        return images_cropped, clust, clu_reg_dets
    else:
        return None, None, None



def get_annotation(phase, image_id, root):
    txt_dir = os.path.join(root,'VisDrone2019-DET-' + phase,'annotations')
    txt_list = glob.glob(txt_dir + "/*.txt")
    txt_list = np.sort(txt_list)
    #print('Find {} samples'.format(len(txt_list)))
    ann_list = []
    for index, line in enumerate(txt_list):
        res = os.path.split(line)
        file_name = res[-1]
        ann_list.append(file_name[:-4])
    #assert image_id in ann_list, "image_id do not in the annotations list"
    ann_path = os.path.join(txt_dir,image_id + '.txt')
    dets = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
    return dets

        
def save_clust_gen_Dets(opt, phase, just_clu_regions, crop_imgs, image_or_path_or_tensor,vis): 
    
    phase = phase
    root = r"H:\Datasets\VisDrone2019"
    image_out_path = os.path.join(root,'VisDrone2019-DET-' + phase,'cluster_'+phase)
    save_txt_dir = os.path.join(root,'VisDrone2019-DET-' + phase,'cluster_'+phase+'_annotations')
    alpha_name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l']
    res = os.path.split(image_or_path_or_tensor)
    image_id = res[1][:6]
    num_reg = 0
    
    reg_list = []
    for regs in just_clu_regions:
        for reg in regs:
            reg_list.append(reg)
            num_reg += 1

    dets = get_annotation(phase, image_id, root)
    dets[:,2] = dets[:,0] + dets[:,2]
    dets[:,3] = dets[:,1] + dets[:,3] # xyxy, scorce, catgriage, _, _

    clu_reg_dets = []
    for i in range(num_reg):
        region = reg_list[i]
        x0 = region[0]
        y0 = region[1]
        x1 = region[2]
        y1 = region[3]
        ct_x = (dets[:,2] + dets[:,0]) / 2 
        ct_y = (dets[:,1] + dets[:,3]) / 2 
        region_dets = dets[np.where((ct_x>=x0) & (ct_x<=x1) & (ct_y>=y0) & (ct_y<=y1))]
        region_dets[:,0] = np.maximum(region_dets[:,0] - x0,0)
        region_dets[:,1] = np.maximum(region_dets[:,1] - y0,0)
        region_dets[:,2] = np.maximum(region_dets[:,2] - x0,0)
        region_dets[:,3] = np.maximum(region_dets[:,3] - y0,0)
        region_dets[:,2] = region_dets[:,2] - region_dets[:,0] 
        region_dets[:,3] = region_dets[:,3] - region_dets[:,1]
        region_dets[:,6] = np.full((len(region_dets)), x0)
        region_dets[:,7] = np.full((len(region_dets)), y0) # xyxy,scorce,cat,clusterx0,clustery0
        # region_dets = region_dets.astype(np.int32)
        if opt.cluster == 1:
            clu_txt_path = os.path.join(save_txt_dir,image_id + str(alpha_name[i]) + ".txt")
            np.savetxt(clu_txt_path,region_dets,fmt='%d',delimiter=',',)
        clu_reg_dets.append(region_dets)
        #画图验证
        if vis:
            import matplotlib.pyplot as plt
            from itertools import cycle
            clu_image_path = os.path.join(image_out_path,image_id + str(alpha_name[i]) + ".jpg")
            # image = cv2.imread(clu_image_path)
        
            plt.figure(1)
            plt.clf()  # 清楚上面的旧图形
            plt.imshow(images_cropped[i][:,:,::-1])
            for k in range(len(region_dets)):
                plt.gca().add_patch(plt.Rectangle(xy=(region_dets[k,0], region_dets[k,1]),
                                    width=region_dets[k,2], #- region_dets[k,0], 
                                    height=region_dets[k,3], #- region_dets[k,1],
                                    edgecolor='r',
                                    fill=False, linewidth=2))
            plt.show()

    if opt.cluster == 2:
        return images_cropped, clust, clu_reg_dets
    else:
        return None, None, None           


"""
if __name__ == "__main__":
    get_annotation(phase='train', image_id='000001')
"""



    









