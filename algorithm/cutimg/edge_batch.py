import os
import cv2
import xlwt
import numpy as np
from skimage import data, filters
import matplotlib.pyplot as plt

'''
输入：
path: 输入的目标背景图像的路径
path_edge: 生成的边缘图像的存储路径
输出:
生成的边缘图像存储于path_edge
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


def caculate(img_t, img_b):
    result1 = sum(sum(sum(img_t))) - sum(sum(sum(img_b)))
    return result1


def main_edge(path, path_edge):
    # path = 'static/images_GLCM'
    # path_edge = 'static/images_GLCM_edge'

    num = 0
    edge_array = [1, 2, 3, 4, 5]
    for filename1 in os.listdir(path):
        if not filename1.startswith('.'):
            path1 = path + '/' + filename1
            path1_edge = path_edge + '/' + filename1

            if os.path.exists(path1_edge) is False:
                os.mkdir(path1_edge)

            for filename2 in os.listdir(path1):
                if not filename2.startswith('.'):
                    path2 = path1 + '/' + filename2
                    path2_edge = path1_edge + '/' + filename2

                    if os.path.exists(path2_edge) is False:
                        os.mkdir(path2_edge)

                    for filename3 in os.listdir(path2):
                        if not filename3.startswith('.'):
                            path3 = path2 + '/' + filename3
                            path3_edge = path2_edge + '/' + filename3

                            # if os.path.exists(path3_edge) is False:
                            #     os.mkdir(path3_edge)

                            path3_edge_canny = path3_edge + '_canny'
                            if os.path.exists(path3_edge_canny) is False:
                                os.mkdir(path3_edge_canny)

                            for img_name in os.listdir(path3):
                                if not img_name.startswith('.'):
                                    num += 1
                                    # print(len(os.listdir(path3)))
                                    img_target = cv2.imread(path3 + '/' + img_name)
                                    if img_target is None:
                                        continue

                                    if img_target.ndim == 3:
                                        img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
                                    elif img_target.ndim == 1:
                                        img_gray = img_target
                                    else:
                                        return 'Please input correct picture!'
                                    img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

                                    img_edge_canny = cv2.Canny(img_target, 50, 150)
                                    cv2.imwrite(path3_edge_canny + '/' + img_name, img_edge_canny)

                                    # cv2.imshow('edge' + img_name, img_edge_canny)
                                    plt.subplot(3, 3, num)
                                    plt.imshow(img_edge_canny, cmap='gray')
                                    plt.title(str(num))
    # cv2.waitKey(0)
    plt.show()
    print(path3_edge_canny)
    return path3_edge_canny

# path_input = 'static\\images_GLCM\\images_camouflage\\mix\\20m'
# main_edge()
