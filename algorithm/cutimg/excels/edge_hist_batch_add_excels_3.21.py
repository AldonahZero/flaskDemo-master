import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import xlwt
import os

'''
输入：
path: 输入目标背景图像的路径
path_edge: 输入目标背景图像对应边缘图像的路径 
path_edge_dh: 边缘方向直方图存储路径
输出:
将边缘方向直方图存储于path_edge_dh
'''
def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

def EDH(rgb_img, TH, edge_canny):
    if rgb_img.ndim == 3:
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    elif rgb_img.ndim == 3:
        return "Please input correct picture!"
    else:
        gray_img = rgb_img
    gray_img = np.array(gray_img)

    if edge_canny.ndim == 3:
        edge_canny = cv2.cvtColor(edge_canny, cv2.COLOR_BGR2GRAY)
    else:
        edge_canny = edge_canny
    edge_canny = np.array(edge_canny)

    row = int(gray_img.shape[0])
    col = int(gray_img.shape[1])
    # gray_img = float(gray_img)
    # 计算梯度矢量Gx, Gy
    Gx = np.zeros((row - 1, col - 1))
    Gy = np.zeros((row - 1, col - 1))
    Gy = np.array(Gy)
    Gx = np.array(Gx)
    theta = np.zeros((row - 1, col - 1))
    theta = np.array(theta)
    Gx = np.zeros((row - 1, col - 1))
    Gy = np.zeros((row - 1, col - 1))
    Gy = np.array(Gy)
    Gx = np.array(Gx)
    theta = np.zeros((row - 1, col - 1))
    theta = np.array(theta)
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            # print(gray_img[i, j])
            Gx1 = gray_img[i - 1, j + 1] + 2 * gray_img[i, j + 1] + gray_img[i + 1, j + 1]
            Gx2 = gray_img[i - 1, j - 1] + 2 * gray_img[i, j - 1] + gray_img[i + 1, j - 1]
            Gy1 = gray_img[i - 1, j - 1] + 2 * gray_img[i - 1, j] + gray_img[i - 1, j + 1]
            Gy2 = gray_img[i + 1, j - 1] + 2 * gray_img[i + 1, j] + gray_img[i + 1, j + 1]
            Gx[i, j] = Gx1 - Gx2
            Gy[i, j] = Gy1 - Gy2
            if Gx[i, j] == 0:
                Gx[i, j] = 1 / np.power(10, 6)
            theta[i, j] = np.arctan2(Gy[i, j], Gx[i, j]) * 180 / math.pi
    bar_hist = np.zeros((1, 36))
    for i in range(0, row - 1):
        for j in range(0, col - 1):
            for k in range(0, 36):
                if (np.int32(edge_canny[i, j]) >= 1) and (np.int32(theta[i, j]) < np.int32(TH[0, k])) and \
                        (np.int32(theta[i, j]) >= np.int32(TH[1, k])):
                    bar_hist[0, k] = bar_hist[0, k] + 1
    bar_hist[0, :] = bar_hist[0, :] / sum(sum(bar_hist))
    bar_hist = np.array(bar_hist)
    return bar_hist

def hist_square(hist_1, hist_2):
    length_of_hist = 36
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
    for i in range(length_of_hist):
        sum1 += hist_1[i]
        sum2 += hist_2[i]
    # 求灰度直方图的均值（期望）
    ex = sum1 / length_of_hist
    ey = sum2 / length_of_hist
    # 求平方的期望
    for i in range(length_of_hist):
        dx += pow(hist_1[i], 2)
        dy += pow(hist_2[i], 2)
    dx = dx / length_of_hist
    dy = dy / length_of_hist
    # 求方差=平方的期望减期望的平方
    dx = dx - pow(ex, 2)
    dy = dy - pow(ey, 2)
    # 求协方差cov=E[XY]-E[X]E[Y]
    for i in range(length_of_hist):
        exy += hist_1[i]*hist_2[i]
    exy = exy / length_of_hist
    cov = exy - ex*ey
    # 求协方差系数 p = cov / (dx开平方*dy开平方)
    p = cov / (pow(dx, 0.5)*pow(dy, 0.5))
    return ex, ey, exy, dx, dy, cov, p, skewness

def main_edge_hist_batch_add_excels():
    TH = [
        [-170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40,
         50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
        [-180, -170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20,
         30,
         40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]]
    TH = np.array(TH)

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('feature', cell_overwrite_ok=True)
    row0 = ["c/nc", "v/i", "h", "angle1_协方差系数", "angle2_协方差系数", "angle3_协方差系数", "angle4_协方差系数",
            "angle5_协方差系数", "angle6_协方差系数", "angle7_协方差系数", "angle8_协方差系数", "angle9_协方差系数",
            "angle10_协方差系数"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    q = 0
    # path = 'D:/chuzhou_data'
    # path_edge = 'D:/chuzhou_data_edge'
    # path_edge_dh = 'D:/chuzhou_data_edge_direct_hist'

    path = 'static\\images_GLCM'
    path_edge = 'static\\images_GLCM_edge'
    path_edge_dh = 'static\\images_GLCM_edge_hist'
    path_excels_save = 'D:/Python/Python/WZ_GLDM/webNew2/static/excels_save_3.21'

    for filename1 in os.listdir(path):
        path1 = path + '/' + filename1
        path1_edge = path_edge + '/' + filename1
        path1_edge_dh = path_edge_dh + '/' + filename1

        if os.path.exists(path1_edge_dh) is False:
            os.mkdir(path1_edge_dh)

        for filename2 in os.listdir(path1):
            path2 = path1 + '/' + filename2
            path2_edge = path1_edge + '/' + filename2
            path2_edge_dh = path1_edge_dh + '/' + filename2

            if os.path.exists(path2_edge_dh) is False:
                os.mkdir(path2_edge_dh)

            for filename3 in os.listdir(path2):
                path3 = path2 + '/' + filename3
                for filename3_edge in os.listdir(path2_edge):
                    path3_edge = path2_edge + '/' + filename3_edge
                    path3_edge_dh = path2_edge_dh + '/' + filename3_edge

                    if os.path.exists(path3_edge_dh) is False:
                        os.mkdir(path3_edge_dh)

                    q = q + 1
                    sheet1.write_merge(q, q, 0, 0, filename1)
                    sheet1.write_merge(q, q, 1, 1, filename2)
                    sheet1.write_merge(q, q, 2, 2, filename3_edge)

                    plt.figure(filename3_edge)
                    for i in range(10):  # 十个角度

                        target_name = str(i) + str(4) + '.jpg'
                        img_target = cv2.imread(path3 + '/' + target_name)
                        img_target_edge = cv2.imread(path3_edge + '/' + target_name)

                        if img_target is None:
                            continue
                        # if img_target.ndim == 3:
                        #     img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

                        # 绘制目标图像边缘方向直方图
                        bar_hist_target = EDH(img_target, TH, img_target_edge)
                        data_pd = np.zeros(36)
                        data_pd[:] = bar_hist_target[0, :]
                        width = 9
                        plt.subplot(3, 3, 5)
                        p3 = plt.bar(TH[0, :], data_pd, width, label="rainfall", color="#87CEFA")
                        # plt.imshow(p3)
                        # 修改为根据前面的路径创建文件夹 存入直方图
                        # plt.savefig(path3_edge_dh + '/' + target_name)

                        if img_target is None:
                            continue
                        [a1, b1, c1] = img_target.shape
                        result = 0
                        cnt = 0
                        sign = True

                        for j in range(9):  # 周围八个背景+目标
                            if j == 4:
                                continue
                            backg_name = str(i) + str(j) + '.jpg'
                            if not os.path.exists(path3 + '/' + backg_name):
                                continue
                            img_backg = cv2.imread(path3 + '/' + backg_name)
                            img_backg_edge = cv2.imread(path3_edge + '/' + backg_name)
                            if img_backg is None:
                                continue

                            # 绘制目标图像边缘方向直方图
                            bar_hist_backg = EDH(img_backg, TH, img_backg_edge)
                            data_pd2 = np.zeros(36)
                            data_pd2[:] = bar_hist_backg[0, :]
                            width = 9
                            plt.subplot(3, 3, j + 1)
                            p2 = plt.bar(TH[0, :], data_pd2, width, label="rainfall", color="#87CEFA")

                            # plt.imshow(p2)
                            # 修改为根据前面的路径创建文件夹 存入直方图
                            # plt.show()
                            # plt.savefig(path3_edge_dh + '/' + backg_name)  #

                            ex1, ey1, exy1, dx1, dy1, cov1, p1, skewness1 = hist_square(data_pd, data_pd2)
                            p1 = np.float32("%.2f" % p1)
                            result = result + p1

                            cnt = cnt + 1
                            sign = False
                        if sign:
                            everage_result = 'NA'
                        else:
                            everage_result = result / cnt
                        print('everage_result = ', everage_result)
                        sheet1.write_merge(q, q, i + 3, i + 3, everage_result)
                        # plt.imshow(p2)

                        plt.savefig(path3_edge_dh + '/' + 'edge_hist.JPG')
                        plt.clf()
    # plt.show()
    f.save(path_excels_save + '/' + 'excel_edge_hist_bacth.xls')
    return path3_edge_dh, everage_result, path_excels_save



main_edge_hist_batch_add_excels()