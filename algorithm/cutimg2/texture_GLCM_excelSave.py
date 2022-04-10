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


def myGLCM_excelSave(path_cutimg, excels_texture_GLCM):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('GLCM', cell_overwrite_ok=True)

    row0 = ["区域名称", "对比度", "同质性", "能量", "相关性", "角二阶矩", "差异性"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    # 修改灰度直方图的方向
    angle = 2  # 0-0° 1-90° 2-180° 3-270°

    target_name = '14.jpg'
    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径
    # excels_texture_GLCM = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/texture_GLCM/'  # GLCM表格存储路径

    img_target = cv2.imread(os.path.join(path_cutimg,target_name))

    # 计算目标图像的灰度共生矩阵值
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    gray_target = img_as_ubyte(gray_target)
    inds = np.digitize(gray_target, bins)
    max_value = inds.max() + 1
    matrix_target = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                 normed=False, symmetric=False)
    sheet1.write_merge(1, 1, 0, 0, "目标区域")
    sheet1.write_merge(1, 1, 1, 1, contrast_feature(matrix_target)[0][angle])
    sheet1.write_merge(1, 1, 2, 2, homogeneity_feature(matrix_target)[0][angle])
    sheet1.write_merge(1, 1, 3, 3, energy_feature(matrix_target)[0][angle])
    sheet1.write_merge(1, 1, 4, 4, correlation_feature(matrix_target)[0][angle])
    sheet1.write_merge(1, 1, 5, 5, asm_feature(matrix_target)[0][angle])
    sheet1.write_merge(1, 1, 6, 6, dissimilarity_feature(matrix_target)[0][angle])
    ##########################

    [a1, b1, c1] = img_target.shape

    result_contrast_0 = 0
    result_dissimilarity_0 = 0
    result_homogeneity_0 = 0
    result_energy_0 = 0
    result_correlation_0 = 0
    result_asm_0 = 0

    cnt = 0
    sign = True
    for j in range(9):
        if j == 4:
            continue
        backg_name = str(1) + str(j) + '.jpg'
        # print(backg_name)
        if not os.path.exists(os.path.join(path_cutimg, backg_name)):
            continue
        img_backg = cv2.imread(os.path.join(path_cutimg, backg_name))
        if img_backg is None:
            continue
        [a2, b2, c2] = img_backg.shape
        # print(backg_name)
        a_min = min(a1, a2)
        b_min = min(b1, b2)
        img_target2 = img_target[0:a_min, 0:b_min, :]
        img_backg2 = img_backg[0:a_min, 0:b_min, :]
        # 修改函数#########################################################################
        gray_bg = cv2.cvtColor(img_backg2, cv2.COLOR_BGR2GRAY)
        gray_bg = img_as_ubyte(gray_bg)
        inds2 = np.digitize(gray_bg, bins)
        max_value2 = inds2.max() + 1
        matrix_bg = greycomatrix(inds2, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                 levels=max_value2,
                                 normed=False, symmetric=False)
        result_contrast_0 = result_contrast_0 + contrast_feature(matrix_bg)[0][angle]
        result_dissimilarity_0 = result_dissimilarity_0 + dissimilarity_feature(matrix_bg)[0][angle]
        result_homogeneity_0 = result_homogeneity_0 + homogeneity_feature(matrix_bg)[0][angle]
        result_energy_0 = result_energy_0 + energy_feature(matrix_bg)[0][angle]
        result_correlation_0 = result_correlation_0 + correlation_feature(matrix_bg)[0][angle]
        result_asm_0 = result_asm_0 + asm_feature(matrix_bg)[0][angle]
        # 以上为修改函数####################################################################
        cnt = cnt + 1
        sign = False
    if sign:
        everage_contrast = 'NA'
        everage_dissimilarity = 'NA'
        everage_homogeneity = 'NA'
        everage_energy = 'NA'
        everage_correlation = 'NA'
        everage_asm = 'NA'
    else:
        everage_contrast = result_contrast_0 / cnt
        everage_dissimilarity = result_dissimilarity_0 / cnt
        everage_homogeneity = result_homogeneity_0 / cnt
        everage_energy = result_energy_0 / cnt
        everage_correlation = result_correlation_0 / cnt
        everage_asm = result_asm_0 / cnt

    sheet1.write_merge(2, 2, 0, 0, "背景区域")
    sheet1.write_merge(2, 2, 1, 1, everage_contrast)
    sheet1.write_merge(2, 2, 2, 2, everage_homogeneity)
    sheet1.write_merge(2, 2, 3, 3, everage_energy)
    sheet1.write_merge(2, 2, 4, 4, everage_correlation)
    sheet1.write_merge(2, 2, 5, 5, everage_asm)
    sheet1.write_merge(2, 2, 6, 6, everage_dissimilarity)
    # print('end')
    res_path = os.path.join(excels_texture_GLCM, 'excel_texture_GLCM.xls')
    f.save(res_path)
    return res_path



