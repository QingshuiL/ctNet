import os 
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth



def meanshift(cluster_dets, image):

    if len(cluster_dets) == 0:
        return None, None
    # 剔除 假阳性目标 
    cluster_dets = cluster_dets[:,:5]
    cluster_dets = cluster_dets[cluster_dets[:,4]>0.1]
    cluster_dets[:,2:4] = cluster_dets[:,2:4] + cluster_dets[:,0:2] - 1
    if len(cluster_dets)==0:
        print('cluster_dets is 0 and should not cluster')
        return None, None
    # cluster_dets = cluster_dets[cluster_dets[:,4]>0.1]
    xc = (cluster_dets[:,0] + cluster_dets[:,2]) / 2
    yc = (cluster_dets[:,1] + cluster_dets[:,3]) / 2
    data = np.concatenate( [xc.reshape(len(xc), 1), yc.reshape(len(yc), 1)], axis=1) 
    # bbox scorce class cx cy
    # 通过下列代码可自动检测bandwidth值
    # 从data中随机选取1000个样本，计算每一对样本的距离，然后选取这些距离的0.2分位数作为返回值，当n_samples很大时，这个函数的计算量是很大的。
    bandwidth = estimate_bandwidth(data, quantile=0.185, n_samples=len(data))
    
    if bandwidth == 0:
        print('bandwidth is 0 and should not cluster')
        return None, None
    # bin_seeding设置为True就不会把所有的点初始化为核心位置，从而加速算法
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    # 计算类别个数
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    #print("number of estimated clusters : %d" % n_clusters)

    clust_reg, n_clusters, cluster_dets = post_clust_region(labels, cluster_dets, n_clusters)

    if clust_reg is None:
        return clust_reg, cluster_dets
    
    # 画图
    pl = 1
    if pl:
        import matplotlib.pyplot as plt
        from itertools import cycle
        
        plt.figure(1)
        plt.clf()  # 清楚上面的旧图形
        plt.imshow(image[:, :, ::-1])
        # cycle把一个序列无限重复下去
        colors = cycle('bgrcmyk')
        for k, color in zip(range(n_clusters), colors):
            # current_member表示标签为k的记为true 反之false
            current_member = labels == k
            cluster_center = cluster_centers[k]
            # 画点
            # plt.plot(data[current_member, 0], data[current_member, 1], color + '.')
            # plt.plot(data[current_member, 0], data[current_member, 1], 'r' + '.')

            # 画圈
            # plt.plot(cluster_center[0], cluster_center[1], 'o',
            #         markerfacecolor=color,  #圈内颜色
            #         markeredgecolor='k',  #圈边颜色
            #         markersize=14)  #圈大小
            
            plt.gca().add_patch(plt.Rectangle(xy=(clust_reg[k,0], clust_reg[k,1]),
                                  width=clust_reg[k,2] - clust_reg[k,0], 
                                  height=clust_reg[k,3] - clust_reg[k,1],
                                  edgecolor=color,
                                  fill=False, linewidth=2))

        plt.ylim(765,0)
        plt.xlim(0,1360)
        plt.title('Estimated number of clusters: %d' % n_clusters)
        plt.show()
        
    # batch = 1
    # clust_reg, clu_dets, n_clusters = select_clu(clust_reg, cluster_dets, n_clusters)

    # if clust_reg is None:
    #     return clust_reg, clu_dets

    # clust_reg = clust_reg.reshape(batch, n_clusters, 5)
    # # clu_dets = clu_dets[:,:6]
    # clusted_dets = clu_dets.reshape(batch,clu_dets.shape[0],clu_dets.shape[1])
    # return clust_reg, clusted_dets

def post_clust_region(labels, cluster_dets, n_clusters): # 108x1 108x7 1
    labels = labels.reshape(len(labels),1).astype(np.int)
    clust = np.concatenate((labels, cluster_dets), axis=1).astype(np.float32)
    clu_list = []
    n = 0
    for i in range(n_clusters):
        cluster = clust[clust[:,0]==i]
        if len(cluster) > 2:
            cluster[:,0] = n
            clu_list.append(cluster)
            n += 1
    n_clusters = n
    if n_clusters == 0:
        clust_reg = None
        cluster_dets = None
        return clust_reg , n_clusters, cluster_dets 

    clust_reg = np.zeros([n_clusters,5]).astype(np.int32)
    cluster_dets = np.concatenate(clu_list, axis=0)

    for i in range(n_clusters):
        x1 = max(0,min(clu_list[i][:,1] )) # 确保不超出256x256边界
        y1 = max(0,min(clu_list[i][:,2] ))
        # x2 = min(255,max(clu_list[i][:,3] + 1))
        # y2 = min(255,max(clu_list[i][:,4] + 1))
        x2 = max(clu_list[i][:,3] )
        y2 = max(clu_list[i][:,4] )
        clust_reg[i] = [x1, y1, x2, y2, i]

    return clust_reg, n_clusters, cluster_dets


if __name__ == "__main__":
    filename = r"D:\BaiduNetdiskDownload\VisDrone2019-DET-train\annotations\000959.txt"
    image_name = r"D:\BaiduNetdiskDownload\VisDrone2019-DET-train\images\000959.jpg"
    dets_list = []
    list_append = dets_list.append

    with open(filename) as file_object:
        lines = file_object.readlines()
        for line in lines:
            bbox = line.strip().split(',')
            bbox = [float(x) for x in bbox]
            # print(bbox)
            list_append(bbox)

    dets = np.vstack(dets_list).reshape(-1,8).astype(np.float32)
    print(dets.shape)
    image = cv2.imread(image_name)
    meanshift(dets, image)










