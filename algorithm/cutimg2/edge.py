import os
import cv2
import xlwt
import numpy as np
from skimage import data,filters
import matplotlib.pyplot as plt


def myEdge(path_cutimg, path_edge, path_edge_canny, path_edge_laplacian, path_edge_log, path_edge_prewitt,
           path_edge_roberts, path_edge_sobel):

    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'
    # path_edge = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/'  # 边缘保存路径
    # path_edge_canny = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/canny/'  # 边缘保存路径
    # path_edge_laplacian = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/laplacian/'  # 边缘保存路径
    # path_edge_log = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/log/'  # 边缘保存路径
    # path_edge_prewitt = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/prewitt/'  # 边缘保存路径
    # path_edge_roberts = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/roberts/'  # 边缘保存路径
    # path_edge_sobel = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/sobel/'  # 边缘保存路径

    resList = []
    for filename in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename):
            continue
        img_target = cv2.imread(os.path.join(path_cutimg, filename))

        img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        img_edge_canny = cv2.Canny(img_target, 50, 150)
        img_edge_canny_result = os.path.join(path_edge_canny, filename)
        cv2.imwrite(img_edge_canny_result, img_edge_canny)
        resList.append(img_edge_canny_result)

        laplacian = cv2.Laplacian(img_target, cv2.CV_16S, ksize=3)
        img_edge_laplacian = cv2.convertScaleAbs(laplacian)
        res_path_edge_laplacian = os.path.join(path_edge_laplacian, filename)
        cv2.imwrite(res_path_edge_laplacian, img_edge_laplacian)
        resList.append(res_path_edge_laplacian)

        x = cv2.Sobel(img_target, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img_target, cv2.CV_16S, 0, 1)
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        img_edge_sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        edge_sobel = os.path.join(path_edge_sobel, filename)
        cv2.imwrite(edge_sobel, img_edge_sobel)
        resList.append(edge_sobel)

        img_edge_roberts = filters.roberts(img_gray)
        path_edge_roberts_result = os.path.join(path_edge_roberts, filename)
        cv2.imwrite(path_edge_roberts_result, img_edge_roberts * 255)
        resList.append(path_edge_roberts_result)

        img_edge_prewitt = filters.prewitt(img_gray)
        path_edge_roberts_result_result = os.path.join(path_edge_prewitt, filename)
        cv2.imwrite(path_edge_roberts_result_result, img_edge_prewitt * 255)
        resList.append(path_edge_roberts_result_result)

        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # img_edge_log = cv2.Laplacian(blurred, -1)
        img_edge_log = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
        path_edge_log_result = os.path.join(path_edge_log, filename)
        cv2.imwrite(path_edge_log_result, img_edge_log)
        resList.append(path_edge_log_result)

    return resList








