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

def myGLDS_excelSave(path_cutimg, excels_texture_GLDS):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('GLDS', cell_overwrite_ok=True)

    row0 = ["区域名称", "均值", "对比度", "角二阶矩", "熵"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径
    # excels_texture_GLDS = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/texture_GLDS/'  # GLDS表格存储路径


    img_target = cv2.imread(path_cutimg + '14.jpg')
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    gray_target = img_as_ubyte(gray_target)
    res_target = get_GLDS(gray_target)
    sheet1.write_merge(1, 1, 0, 0, "目标区域")
    sheet1.write_merge(1, 1, 1, 1, res_target[0])
    sheet1.write_merge(1, 1, 2, 2, res_target[1])
    sheet1.write_merge(1, 1, 3, 3, res_target[2])
    sheet1.write_merge(1, 1, 4, 4, res_target[3])

    [a1, b1, c1] = img_target.shape

    result_1 = 0
    result_2 = 0
    result_3 = 0
    result_4 = 0

    cnt = 0
    sign = True
    for j in range(9):
        if j == 4:
            continue
        backg_name = str(1) + str(j) + '.jpg'
        # print(backg_name)
        if not os.path.exists(path_cutimg + backg_name):
            continue
        img_backg = cv2.imread(path_cutimg + backg_name)
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

        res_bg = get_GLDS(gray_bg)
        result_1 = result_1 + res_bg[0]
        result_2 = result_2 + res_bg[1]
        result_3 = result_3 + res_bg[2]
        result_4 = result_4 + res_bg[3]
        # 以上为修改函数####################################################################
        cnt = cnt + 1
        sign = False
    if sign:
        everage_1 = 'NA'
        everage_2 = 'NA'
        everage_3 = 'NA'
        everage_4 = 'NA'
    else:
        everage_1 = result_1 / cnt
        everage_2 = result_2 / cnt
        everage_3 = result_3 / cnt
        everage_4 = result_4 / cnt
    sheet1.write_merge(2, 2, 0, 0, "背景区域")
    sheet1.write_merge(2, 2, 1, 1, everage_1)
    sheet1.write_merge(2, 2, 2, 2, everage_2)
    sheet1.write_merge(2, 2, 3, 3, everage_3)
    sheet1.write_merge(2, 2, 4, 4, everage_4)

    f.save(excels_texture_GLDS + 'excel_texture_GLDS.xls')

    return excels_texture_GLDS

