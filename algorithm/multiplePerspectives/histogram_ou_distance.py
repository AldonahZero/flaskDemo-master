import numpy as np
import cv2
import os
import math

def grey_distance(img1, img2):  #计算两幅图的灰度直方图距离
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #img1 = cv2.equalizeHist(img1)
    #img2 = cv2.equalizeHist(img2)

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 255])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 255])

    m = len(img1.nonzero()[0])
    n = len(img2.nonzero()[0])
    for i in range(0, 256):
        hist1[i] = hist1[i] / m
        hist2[i] = hist2[i] / n
    oud = 0
    for i in range(0, 256):
        oud = oud + pow(hist1[i] - hist2[i], 2)
        #oud = oud + abs(hist1[i] - hist2[i])
    oud = np.sqrt(oud)
    return oud

if __name__ == '__main__':  # 测试灰度直方图
    path1 = 'static/images/lhy/001.jpg'
    path2 = 'static/images/lhy/002.jpg'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    res = grey_distance(img1, img2)
    print(res)



