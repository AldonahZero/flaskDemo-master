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


def myGrayMean_excelSave(path_cutimg, excels_color_gray_mean):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('灰度均值', cell_overwrite_ok=True)

    row0 = ["灰度均值", "背景1", "背景2", "背景3", "背景4", "目标", "背景6", "背景7", "背景8", "背景9"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    q = 1

    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径
    # excels_color_gray_mean = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/color_gray_mean/'

    for filename1 in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename1):
            q += 1
            continue

        img = cv2.imread(path_cutimg + '/' + filename1, 0);
        height, width = img.shape
        size = img.size

        average = 0
        for i in range(height):
            for j in range(width):
                average += img[i][j] / size

        sheet1.write_merge(1, 1, q, q, average)
        q += 1
    excel_save_path = os.path.join(excels_color_gray_mean, 'excel_color_gray_mean.xls')

    f.save(excel_save_path )
    return excel_save_path

