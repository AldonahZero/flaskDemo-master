import cv2
import numpy as np
import os
import xlwt
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

# 设置表格样式


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

def get_GLDS(img1):
    result = []

    img = np.array(img1).astype(np.float64)
    m, n = img.shape
    # print(img.shape)

    B = img.copy()

    C = np.zeros((m, n))

    for i in range(m - 1):
        for j in range(n - 1):
            # B[i,j]= img[i + 1, j + 1]
            a1 = float(img[i + 1, j + 1])
            a2 = float(img[i, j])
            C[i, j] = abs(round(a1 - a2))
    mn = np.min(C)
    mx = np.max(C)
    norm_C = (C - mn) * (1.0 / (mx - mn))

    hist, bins = np.histogram(norm_C.flatten(), bins=256)
    # hist_cv=cv2.calcHist([norm_C],[0],None,[256],[0,256])
    hist_re = hist / (m * n)

    MEAN = 0  # 均值
    CON = 0  # 对比度
    ASM = 0  # 角二阶矩
    ENT = 0  # 熵

    for i in range(256):
        MEAN = MEAN + (i * hist_re[i]) / 256
        CON = CON + i * i * hist_re[i]
        ASM = ASM + hist_re[i] * hist_re[i]
        if (hist_re[i] > 0):
            ENT = ENT - hist_re[i] * np.log2(hist_re[i])
    # result[0] = MEAN
    # result[1] = CON
    # result[2] = ASM
    # result[3] = ENT
    return [MEAN, CON, ASM, ENT]


def myGLDS(path_cutimg):

    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径

    img_target = cv2.imread(path_cutimg + '14.jpg')

    # 计算目标图像的灰度共生矩阵值
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    gray_target = img_as_ubyte(gray_target)

    res_target = get_GLDS(gray_target)

    return res_target
