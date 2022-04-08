import numpy as np
import cv2
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


# GLCM properties


def contrast_feature(matrix_coocurrence):
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    # return "Contrast = ", contrast
    return contrast


def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
    return dissimilarity


def homogeneity_feature(matrix_coocurrence):
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity


def energy_feature(matrix_coocurrence):
    energy = greycoprops(matrix_coocurrence, 'energy')
    return energy


def correlation_feature(matrix_coocurrence):
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    return correlation


def asm_feature(matrix_coocurrence):
    asm = greycoprops(matrix_coocurrence, 'ASM')
    return asm


def myGLCM(path_cutimg):

    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    # 修改灰度直方图的方向
    angle = 2  # 0-0° 1-90° 2-180° 3-270°

    img_target = cv2.imread(path_cutimg + '14.jpg')

    # 计算目标图像的灰度共生矩阵值
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    gray_target = img_as_ubyte(gray_target)
    inds = np.digitize(gray_target, bins)
    max_value = inds.max() + 1
    matrix_target = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                 normed=False, symmetric=False)
    arr = np.zeros(6)
    arr[0] = contrast_feature(matrix_target)[0][angle]
    arr[1] = homogeneity_feature(matrix_target)[0][angle]
    arr[2] = energy_feature(matrix_target)[0][angle]
    arr[3] = correlation_feature(matrix_target)[0][angle]
    arr[4] = asm_feature(matrix_target)[0][angle]
    arr[5] = dissimilarity_feature(matrix_target)[0][angle]

    return arr




