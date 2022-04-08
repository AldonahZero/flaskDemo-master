import os
import cv2
import xlwt
import matplotlib.pyplot as plt
import math
import numpy as np


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def hist_square(hist_1, hist_2):
    sum1 = 0
    sum2 = 0
    ex = 0
    ey = 0
    exy = 0
    dx = 0
    dy = 0
    cov = 0
    skewness = 0  # 目标图像的偏斜度，代表目标图像分布的位置
    # 协方差系数
    p = 0
    for i in range(0, 255):
        sum1 += hist_1[i]
        sum2 += hist_2[i]
    # 求灰度直方图的均值（期望）
    ex = sum1 / 255
    ey = sum2 / 255
    # 求平方的期望
    for i in range(0, 255):
        dx += pow(hist_1[i], 2)
        dy += pow(hist_2[i], 2)
    dx = dx / 255
    dy = dy / 255
    # 求方差=平方的期望减期望的平方
    dx = dx - pow(ex, 2)
    dy = dy - pow(ey, 2)
    # 求协方差cov=E[XY]-E[X]E[Y]
    for i in range(0, 255):
        exy += hist_1[i] * hist_2[i]
    exy = exy / 255
    cov = exy - ex * ey
    # 求协方差系数 p = cov / (dx开平方*dy开平方)
    p = cov / (pow(dx, 0.5) * pow(dy, 0.5))
    return ex, ey, exy, dx, dy, cov, p, skewness

def myGrayHitogram_excelSave(path_original, path_bitwise, excels_color_gray_histogram):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('灰度直方图', cell_overwrite_ok=True)

    row0 = ["目标背景灰度直方图相关性", "协方差", "协方差系数"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    q = 0

    # path_original = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_original/1.jpg'
    # path_bitwise = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_bitwise/1.jpg'
    # excels_color_gray_histogram = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/color_gray_histogram/'

    array_t_length = 0
    array_b_length = 0

    bitwise = cv2.imread(path_bitwise, 0)

    gray_original = cv2.imread(path_original, 0)

    [height, width] = bitwise.shape
    for i in range(height - 1):
        for j in range(width - 1):
            pixel_value2 = bitwise[i, j]
            if pixel_value2 == 0:
                array_b_length = array_b_length + 1
            else:
                array_t_length = array_t_length + 1
    array_t = [0] * array_t_length
    array_b = [0] * array_b_length

    t = 0
    b = 0
    A = gray_original

    for i in range(height - 1):
        for j in range(width - 1):
            pixel_value = A[i, j]
            pixel_value2 = bitwise[i, j]
            if pixel_value2 == 0:
                array_b[b] = pixel_value
                # print('b, pixel_value = ', b, pixel_value)
                b = b + 1

            else:
                # print('t = ', t)
                array_t[t] = pixel_value
                t = t + 1
    array_t = np.array(array_t, np.float32)
    array_b = np.array(array_b, np.float32)

    hist_t = cv2.calcHist([array_t], [0], None, [256], [0, 255])
    hist_b = cv2.calcHist([array_b], [0], None, [256], [0, 255])

    ex1, ey1, exy1, dx1, dy1, cov1, p1, skewness1 = hist_square(hist_t, hist_b)
    cov1 = "%.2f" % cov1
    p1 = "%.2f" % p1

    sheet1.write_merge(1, 1, 1, 1, cov1)
    sheet1.write_merge(1, 1, 2, 2, p1)



    excel_save_path = os.path.join(excels_color_gray_histogram, 'excel_color_gray_histogram.xls')

    f.save(excel_save_path )
    return excel_save_path

# print(myGrayHitogram_excelSave())



