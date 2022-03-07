import numpy as np
import math
import algorithm.HSI as hsi


# 输入 是图片路径 和 所选择波段的数量  输出是对应的整数数组
def ECA_f(image_path, k_num):
    image = hsi.load_data(image_path)
    [_, _, p] = image.shape
    image_2d = hsi.to_2d_array(image)
    dij = np.zeros((p, p), float)
    for i in range(p):
        for j in range(i, p):
            dij[i][j] = np.sqrt(np.sum(np.square(image_2d[i][:] - image_2d[j][:])))
            dij[j][i] = dij[i][j]
            #  dist = np.linalg.norm(vec1 - vec2)
    sums = np.zeros(p, float)
    for i in range(p):
        for j in range(p):
            sums[i] = sums[i] + math.exp(-(dij[i][j]))
    sort = np.argsort(sums)
    B = np.zeros(p, float)
    for i in range(p - 1):
        b = sort[i]
        c = sort[i + 1]
        B[b] = dij[b][c]
    b = sort[p - 1]
    B[b] = dij[b].max()
    sums2 = B * sums
    res = np.argsort(sums2)
    res = res[::-1]
    result = np.zeros(k_num, int)
    for i in range(k_num):
        result[i] = res[i] + 1
    return result