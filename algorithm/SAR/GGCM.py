import csv
import cv2
import os
import numpy as np

from config.setting import RESULT_FOLDER
from .process_pre import nsst_dec

np.set_printoptions(suppress=True)  # 输出时禁止科学表示法，直接输出小数值


def ggcm_features(mat, ngrad=16, ngray=16):
    '''
    :param mat: 灰度梯度共生矩阵
    :return: 由共生矩阵的特征参数
    '''
    H_colsum = sum(mat)
    H_sumrow = sum(mat.T)
    total = mat.sum()
    H_p = mat / total
    # 小梯度优势 T1
    T1 = 0
    for j in range(ngrad):
        T1 += H_colsum[j] / ((j + 1) ** 2)
    T1 = T1 / total
    # 大梯度优势 T2
    T2 = 0
    for j in range(ngrad):
        T2 += H_colsum[j] * (j + 1) ** 2
    T2 = T2 / total
    # 灰度分布不均匀性 T3
    T3 = 0
    for j in range(ngray):
        T3 += H_sumrow[j] ** 2
    T3 = T3 / total
    # 梯度分布的不均匀性 T4
    T4 = 0
    for j in range(ngrad):
        T4 += H_colsum[j] ** 2
    T4 = T4 / total
    # 能量 T5
    T5 = 0
    for i in range(ngray):
        for j in range(ngrad):
            T5 += H_p[i][j] ** 2
    # 灰度均值 T6
    T6 = 0
    H_psumrow = sum(H_p.T)
    for i in range(ngray):
        T6 += (j + 1) * H_psumrow[j]
    # 梯度均值 T7
    T7 = 0
    H_psumcol = sum(H_p)
    for j in range(ngrad):
        T7 += (j + 1) * H_psumcol[j]
    # 灰度标准差 T8
    T8 = 0
    for j in range(ngray):
        T8 += (j + 1 - T6) ** 2 * H_psumrow[j]
    T8 = T8 ** 0.5
    # 梯度标准差 T9
    T9 = 0
    for j in range(ngrad):
        T9 += (j + 1 - T7) ** 2 * H_psumcol[j]
    T9 = T9 ** 0.5
    # 相关 T10
    T10 = 0
    for i in range(ngray):
        for j in range(ngrad):
            T10 += (i + 1 - T6) * (j + 1 - T7) * H_p[i][j]
    # 灰度熵 T11
    T11 = 0
    for j in range(ngray):
        T11 += H_psumrow[j] * np.log10(H_psumrow[j] + 2e-16)
    T11 = -T11
    # 梯度熵 T12
    T12 = 0
    for j in range(ngrad):
        T12 += H_psumcol[j] * np.log10(H_psumcol[j] + 2e-16)
    T12 = -T12
    # 混合熵 T13
    T13 = 0
    for i in range(ngray):
        for j in range(ngrad):
            T13 += H_p[i][j] * np.log10(H_p[i][j] + 2e-16)
    T13 = -T13
    # 惯性 T14
    T14 = 0
    for i in range(ngray):
        for j in range(ngrad):
            T14 += (i - j) ** 2 * H_p[i][j]
    # 逆差矩 T15
    T15 = 0
    for i in range(ngray):
        for j in range(ngrad):
            T15 += H_p[i, j] / (1 + (i - j) ** 2)
    ggcm_features = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15]
    return ggcm_features


def get_ggcm_features(path, ngrad=16, ngray=16):
    '''
    :param img: 输入图像
    :param ngrad:灰度图压缩
    :param ngray:梯度图压缩
    :return:glgcm_features_path
    '''
    _, gray_img = nsst_dec(path)
    img = cv2.copyMakeBorder(gray_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # 图像扩展
    [m, n] = img.shape
    print(m)
    img_gray = img  # 扩展后图像
    img_gray1 = img[1:m - 1, 1:n - 1]  # 原图
    # [x,y] = img_gray.shape
    gsx = np.zeros([m - 2, n - 2])
    gsy = np.zeros([m - 2, n - 2])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            gsx[i - 1, j - 1] = -1 * img_gray[i - 1][j - 1] - 2 * img_gray[i - 1][j] - 1 * img_gray[i - 1][j + 1] + 1 * \
                                img_gray[i + 1][j - 1] + 2 * img_gray[i + 1][j] + 1 * img_gray[i + 1][j + 1]
            gsy[i - 1][j - 1] = -1 * img_gray[i - 1][j - 1] - 2 * img_gray[i][j - 1] - 1 * img_gray[i + 1][j - 1] + 1 * \
                                img_gray[i - 1][j + 1] + 2 * img_gray[i][j + 1] + 1 * img_gray[i + 1][j + 1]
    grad = (gsx ** 2 + gsy ** 2) ** 0.5;  # 计算梯度值

    grad = np.asarray(1.0 * grad * (ngrad - 1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * img_gray1 * (ngray - 1) / img_gray1.max(), dtype=np.int16)  # 0-255变换为0-15
    # 梯度归一化，灰度归一化

    gray_grad = np.zeros([ngray, ngrad])  # 灰度梯度共生矩阵
    for i in range(m - 2):
        for j in range(n - 2):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad = 1.0 * gray_grad / (m * n)  # 归一化灰度梯度矩阵，减少计算量
    ggcm_feature = ggcm_features(gray_grad, ngrad=16, ngray=16)
    # ggcm_features_path = RESULT_FOLDER + '/SAR/ggcm.csv'
    ggcm_features_path = os.path.join(RESULT_FOLDER, 'SAR/ggcm.csv')
    f = open(ggcm_features_path, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    # 构建列表头
    csv_writer.writerow(
        ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15'])
    # 写入csv文件内容
    csv_writer.writerow(ggcm_feature)
    f.close()
    return ggcm_features_path, ggcm_feature

# if __name__ == '__main__':
#     glgcm_features = get_ggcm_features('../../' +'static/result'+ '/SAR/image_f.png')
#     print(glgcm_features)
