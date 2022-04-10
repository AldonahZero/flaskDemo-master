import os
import cv2
import xlwt
import numpy as np
from skimage import data,filters
import matplotlib.pyplot as plt


# def myEdge(path_cutimg, path_edge, path_edge_canny, path_edge_laplacian, path_edge_log, path_edge_prewitt,
#            path_edge_roberts, path_edge_sobel):
#
#     for filename in os.listdir(path_cutimg):
#         if not os.path.exists(path_cutimg + '/' + filename):
#             continue
#         img_target = cv2.imread(path_cutimg + filename)
#
#         img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
#
#         img_edge_canny = cv2.Canny(img_target, 50, 150)
#         cv2.imwrite(path_edge_canny + filename, img_edge_canny)
#
#         laplacian = cv2.Laplacian(img_target, cv2.CV_16S, ksize=3)
#         img_edge_laplacian = cv2.convertScaleAbs(laplacian)
#         cv2.imwrite(path_edge_laplacian + filename, img_edge_laplacian)
#
#         x = cv2.Sobel(img_target, cv2.CV_16S, 1, 0)
#         y = cv2.Sobel(img_target, cv2.CV_16S, 0, 1)
#         Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
#         Scale_absY = cv2.convertScaleAbs(y)
#         img_edge_sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
#         cv2.imwrite(path_edge_sobel + filename, img_edge_sobel)
#
#         img_edge_roberts = filters.roberts(img_gray)
#         cv2.imwrite(path_edge_roberts + filename, img_edge_roberts * 255)
#
#         img_edge_prewitt = filters.prewitt(img_gray)
#         cv2.imwrite(path_edge_prewitt + filename, img_edge_prewitt * 255)
#
#         blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
#         # img_edge_log = cv2.Laplacian(blurred, -1)
#         img_edge_log = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
#         cv2.imwrite(path_edge_log + filename, img_edge_log)
#
#     return path_edge

def myEdgeCanny(path_cutimg, path_edge, path_edge_canny):
    for filename in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename):
            continue
        img_target = cv2.imread(path_cutimg + filename)

        img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        img_edge_canny = cv2.Canny(img_target, 50, 150)
        cv2.imwrite(path_edge_canny + filename, img_edge_canny)

    return path_edge_canny


def myEdgeLaplacian(path_cutimg, path_edge, path_edge_laplacian):
    for filename in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename):
            continue
        img_target = cv2.imread(path_cutimg + filename)

        img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(img_target, cv2.CV_16S, ksize=3)
        img_edge_laplacian = cv2.convertScaleAbs(laplacian)
        cv2.imwrite(path_edge_laplacian + filename, img_edge_laplacian)

    return path_edge_laplacian


def myEdgeSobel(path_cutimg, path_edge, path_edge_sobel):
    for filename in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename):
            continue
        img_target = cv2.imread(path_cutimg + filename)

        img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        x = cv2.Sobel(img_target, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img_target, cv2.CV_16S, 0, 1)
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        img_edge_sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        cv2.imwrite(path_edge_sobel + filename, img_edge_sobel)

    return path_edge_sobel


def myEdgeRoberts(path_cutimg, path_edge, path_edge_roberts):
    for filename in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename):
            continue
        img_target = cv2.imread(path_cutimg + filename)

        img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        img_edge_roberts = filters.roberts(img_gray)
        cv2.imwrite(path_edge_roberts + filename, img_edge_roberts * 255)

    return path_edge_roberts


def myEdgePrewitt(path_cutimg, path_edge, path_edge_prewitt):
    for filename in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename):
            continue
        img_target = cv2.imread(path_cutimg + filename)

        img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        img_edge_prewitt = filters.prewitt(img_gray)
        cv2.imwrite(path_edge_prewitt + filename, img_edge_prewitt * 255)

    return path_edge_prewitt


def myEdgeLog(path_cutimg, path_edge, path_edge_log):
    for filename in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename):
            continue
        img_target = cv2.imread(path_cutimg + filename)

        img_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # img_edge_log = cv2.Laplacian(blurred, -1)
        img_edge_log = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
        cv2.imwrite(path_edge_log + filename, img_edge_log)

    return path_edge_log