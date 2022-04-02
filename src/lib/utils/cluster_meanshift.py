import numpy as np
import math
 
MIN_DISTANCE = 0.00001  # 最小误差
 
 
def euclidean_dist(pointA, pointB):
    # 计算pointA和pointB之间的欧式距离
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)
 
 
def gaussian_kernel(distance, bandwidth):
    ''' 高斯核函数
    :param distance: 欧氏距离计算函数
    :param bandwidth: 核函数的带宽
    :return: 高斯函数值
    '''
    m = np.shape(distance)[0]  # 样本个数
    right = np.mat(np.zeros((m, 1)))
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))
    gaussian_val = left * right
    return gaussian_val
 
 
def shift_point(point, points, kernel_bandwidth):
    '''计算均值漂移点
    :param point: 需要计算的点
    :param points: 所有的样本点
    :param kernel_bandwidth: 核函数的带宽
    :return:
        point_shifted：漂移后的点
    '''
    points = np.mat(points)
    m = np.shape(points)[0]  # 样本个数
    # 计算距离
    point_distances = np.mat(np.zeros((m, 1)))
    for i in range(m):
        point_distances[i, 0] = euclidean_dist(point, points[i])
 
    # 计算高斯核
    point_weights = gaussian_kernel(point_distances, kernel_bandwidth)
 
    # 计算分母
    all = 0.0
    for i in range(m):
        all += point_weights[i, 0]
 
    # 均值偏移
    point_shifted = point_weights.T * points / all
    return point_shifted
 
 
def group_points(mean_shift_points):
    '''计算所属的类别
    :param mean_shift_points:漂移向量
    :return: group_assignment：所属类别
    '''
    group_assignment = []
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
 
        item_1 = "_".join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1
 
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
 
        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])
    return group_assignment
 
 
def train_mean_shift(points, kernel_bandwidth=2):
    '''训练Mean Shift模型
    :param points: 特征数据
    :param kernel_bandwidth: 核函数带宽
    :return:
        points：特征点
        mean_shift_points：均值漂移点
        group：类别
    '''
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0
    m = np.shape(mean_shift_points)[0]  # 样本的个数
    need_shift = [True] * m  # 标记是否需要漂移
 
    # 计算均值漂移向量
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iteration += 1
        print("iteration : " + str(iteration))
        for i in range(0, m):
            # 判断每一个样本点是否需要计算偏置均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kernel_bandwidth)  # 对样本点进行偏移
            dist = euclidean_dist(p_new, p_new_start)  # 计算该点与漂移后的点之间的距离
 
            if dist > max_min_dist:  # 记录是有点的最大距离
                max_min_dist = dist
            if dist < MIN_DISTANCE:  # 不需要移动
                need_shift[i] = False
 
            mean_shift_points[i] = p_new
    # 计算最终的group
    group = group_points(mean_shift_points)  # 计算所属的类别
    return np.mat(points), mean_shift_points, group