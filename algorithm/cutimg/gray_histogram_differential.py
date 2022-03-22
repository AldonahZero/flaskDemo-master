import os
import cv2
import xlwt
import matplotlib
from config.setting import MATPLOTLIB_INSHOW
if not MATPLOTLIB_INSHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np

'''
输入：
path:原始图像的输入路径
path_bitwise:掩膜图像的输入路径
path_gray_histogram_save:目标与背景灰度直方图的存储路径
输出：
目标背景的灰度直方图会存储于path_gray_histogram_save
并在下方输出目标背景灰度直方图对应的 协方差cov 以及协方差系数p
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

def plt_hist(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

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

def main_gray_hist_differential(path,path_bitwise,path_gray_histogram_save):

    q = 0

    # path = 'static/images_GLCM_original'
    # path_bitwise = 'static/images_GLCM_bitwise'
    # path_gray_histogram_save = 'static/images_save/gray_histogram/'

    # path = path_input + '_original'
    # path_bitwise = path_input + '_bitwise'
    for filename1 in os.listdir(path):
        if filename1.startswith('.'): continue
        path1 = path + '/' + filename1
        path1_bitwise = path_bitwise + '/' + filename1
        for filename2 in os.listdir(path1):
            if filename2.startswith('.'): continue
            path2 = path1 + '/' + filename2
            path2_bitwise = path1_bitwise + '/' + filename2
            for filename3 in os.listdir(path2):
                if filename3.startswith('.'): continue
                path3 = path2 + '/' + filename3
                # path3_bitwise = path2_bitwise + '/' + 'bitwise' + filename3
                path3_bitwise = path2_bitwise + '/' + filename3
                # print(path3_bitwise)

                q = q + 1
                # sheet1.write_merge(q, q, 0, 0, filename1)
                # sheet1.write_merge(q, q, 1, 1, filename2)
                # sheet1.write_merge(q, q, 2, 2, filename3)
                img_name = 0
                # bitwise_name = 0

                # 用list代替数组(最快)
                for original_name in os.listdir(path3):
                    # print(original_name)
                    array_t_length = 0
                    array_b_length = 0

                    # 如果原图为空
                    if not os.path.exists(path3 + '/' + original_name):
                        img_name = img_name + 1
                        # print(0)
                        continue
                    # if not os.path.exists(path3_bitwise + '/' + str(img_name) + '.JPG'):
                    if not os.path.exists(path3_bitwise + '/' + original_name):
                        # print(img_name)
                        img_name = img_name + 1
                        # print(img_name)
                        continue

                    img_original = cv2.imread(path3 + '/' + original_name)
                    # print(2)
                    bitwise = cv2.imread(path3_bitwise + '/' + original_name)
                    bitwise = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)
                    # print('path of o', path6 + '/' + original_name)
                    gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
                    # print(bitwise.shape)
                    # print(gray_original.shape)
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
                    # print(len(array_t), len(array_b))
                    t = 0
                    b = 0
                    # 取差分图像 ####### begin
                    # A = np.zeros((height, width))
                    # for i in range(height - 1):
                    #     for j in range(width - 1):
                    #         A[i, j] = math.fabs(int(gray_original[i, j]) - int(gray_original[i + 1, j + 1]))
                    A = gray_original
                    ################## end
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

                    # plt.figure(1)
                    # plt.hist(array_t, bins=256, range=None)
                    # plt.figure(2)
                    # plt.hist(array_b, bins=256, range=None)
                    # plt.show()
                    # print(array_t, len(array_t))
                    # print(array_b, len(array_b))
                    # print(max(array_t))
                    # print(max(array_b))
                    hist_t = cv2.calcHist([array_t], [0], None, [256], [0, 255])
                    hist_b = cv2.calcHist([array_b], [0], None, [256], [0, 255])
                    # print(len(hist_t))
                    # print(len(hist_b))

                    ex1, ey1, exy1, dx1, dy1, cov1, p1, skewness1 = hist_square(hist_t, hist_b)
                    cov1 = "%.2f" % cov1
                    p1 = "%.2f" % p1

                    plt.figure('ρ = ' + str(p1),figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.title("Grayscale Histogram of Target" + '  ' + 'ρ = ' + str(p1))
                    plt.xlabel("Gray Levels")
                    plt.ylabel("Numbers of Pixel")
                    # plt.hist(hist_t, bins=256,density=None,facecolor='b',alpha=1,histtype='bar')
                    # plt.xlim([0, 256])
                    # plt.hist(hist_t, 256, [0, 256])
                    plt.hist(array_t, bins=256, range=None)

                    plt.subplot(1, 2, 2)
                    # plt.figure('background')
                    plt.title("Grayscale Histogram of Background")
                    plt.xlabel("Gray Levels")
                    # plt.ylabel("Numbers of Pixel")
                    # plt.hist(hist_b, bins=256,density=None,facecolor='b',alpha=1,histtype='bar')
                    # plt.xlim([0, 256])
                    # plt.hist(hist_b, 256, [0, 256])
                    plt.hist(array_b, bins=256, range=None)
                    '''
                    plt.savefig('static\\images_save\\gray_histogram\\' + original_name)
                    '''
                    print(path_gray_histogram_save + original_name)
                    plt.savefig(path_gray_histogram_save + original_name)

                    # plt.show()

                    # print('cov, p = ', cov1, p1)

                    img_name = img_name + 1
                # path_mid = path_gray_histogram_save + original_name
                return path_gray_histogram_save



    # f.save('static\\excel_save\\gray_histogram_diff.xls')
# main_gray_hist_differential()