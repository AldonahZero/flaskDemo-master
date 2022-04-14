import cv2
import os
import matplotlib.pyplot as plt
from .process_pre import nsst_dec
from config.setting import RESULT_FOLDER


def peak_value(path1):
    '''
    :param path1: 原图路径
    :return: prek_image_path: 峰值点图存储路径
    '''
    img = cv2.imread(path1)
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path,img_f = nsst_dec(path1)
    print(path)
    img1 = img_f / img_f.max()
    m, n = img1.shape
    thgma = 0.1
    u = img1.sum() / (m * n)
    k_t = 2
    # 设置门限，去除部分杂波处可能检测到的峰值
    for i in range(m):
        for k in range(n):
            if img1[i][k] < u + k_t * thgma:
                img1[i][k] = 0
    plt.figure()
    plt.imshow(img, plt.cm.gray)
    # 按照定义求峰值点
    for i in range(2, m - 1):
        for k in range(2, n - 1):
            cell = img1[i - 2:i + 1, k - 2:k + 1]
            cell_q = cell[1][1] - cell
            q_min = cell_q.min()
            b = cell_q != 0
            if b.sum() == 8:
                if q_min >= 0:
                    plt.scatter(k - 1, i - 1, c='red', marker='*')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('峰值点')
    # prek_image_path = RESULT_FOLDER + '/SAR/peak_value.png'
    prek_image_path = os.path.join(RESULT_FOLDER,'SAR/peak_value.png')
    plt.savefig(prek_image_path)
    return prek_image_path

# if __name__ == '__main__':
#     print(peak_value())
