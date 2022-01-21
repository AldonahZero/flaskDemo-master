import os
import cv2
import xlwt
import numpy as np
from skimage import data,filters
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

def main_edge():

    path = 'static\\images_GLCM'
    path_edge = 'static\\images_GLCM_edge'

    num = 0
    edge_array = [1, 2, 3, 4, 5]
    for filename1 in os.listdir(path):
        path1 = path + '/' + filename1
        path1_edge = path_edge + '/' + filename1

        if os.path.exists(path1_edge) is False:
            os.mkdir(path1_edge)

        for filename2 in os.listdir(path1):
            path2 = path1 + '/' + filename2
            path2_edge = path1_edge + '/' + filename2

            if os.path.exists(path2_edge) is False:
                os.mkdir(path2_edge)

            for filename3 in os.listdir(path2):
                path3 = path2 + '/' + filename3
                path3_edge = path2_edge + '/' + filename3

                # if os.path.exists(path3_edge) is False:
                #     os.mkdir(path3_edge)

                path3_edge_canny = path3_edge + '_canny'
                if os.path.exists(path3_edge_canny) is False:
                    os.mkdir(path3_edge_canny)
                '''    
                path3_edge_laplacian = path3_edge + '_laplacian'
                if os.path.exists(path3_edge_laplacian) is False:
                    os.mkdir(path3_edge_laplacian)

                path3_edge_sobel = path3_edge + '_sobel'
                if os.path.exists(path3_edge_sobel) is False:
                    os.mkdir(path3_edge_sobel)

                path3_edge_roberts = path3_edge + '_roberts'
                if os.path.exists(path3_edge_roberts) is False:
                    os.mkdir(path3_edge_roberts)

                path3_edge_prewitt = path3_edge + '_prewitt'
                if os.path.exists(path3_edge_prewitt) is False:
                    os.mkdir(path3_edge_prewitt)

                path3_edge_log = path3_edge + '_log'
                if os.path.exists(path3_edge_log) is False:
                    os.mkdir(path3_edge_log)
                '''

                for img_name in os.listdir(path3):
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

                    '''
                    laplacian = cv2.Laplacian(img_target, cv2.CV_16S, ksize=3)
                    img_edge_laplacian = cv2.convertScaleAbs(laplacian)
                    cv2.imwrite(path3_edge_laplacian + '/' + img_name, img_edge_laplacian)

                    x = cv2.Sobel(img_target, cv2.CV_16S, 1, 0)
                    y = cv2.Sobel(img_target, cv2.CV_16S, 0, 1)
                    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
                    Scale_absY = cv2.convertScaleAbs(y)
                    img_edge_sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
                    cv2.imwrite(path3_edge_sobel + '/' + img_name, img_edge_sobel)

                    img_edge_roberts = filters.roberts(img_gray)
                    cv2.imwrite(path3_edge_roberts + '/' + img_name, img_edge_roberts)

                    img_edge_prewitt = filters.prewitt(img_gray)
                    cv2.imwrite(path3_edge_prewitt + '/' + img_name, img_edge_prewitt)

                    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
                    img_edge_log = cv2.Laplacian(blurred, -1)
                    cv2.imwrite(path3_edge_log + '/' + img_name, img_edge_log)
                    '''

                    # cv2.imshow('edge' + img_name, img_edge_canny)
                    plt.subplot(3, 3, num)
                    plt.imshow(img_edge_canny, cmap='gray')
                    plt.title(str(num))
    # cv2.waitKey(0)
    plt.show()
    return path3_edge_canny


# path_input = 'static\\images_GLCM\\images_camouflage\\mix\\20m'
# main_edge()
