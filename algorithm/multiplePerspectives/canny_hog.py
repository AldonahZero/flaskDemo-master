import numpy as np
import cv2
import os
import skimage
import math
from skimage.feature import hog

def gcd(a, b): # 求两数最大公约数
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    return a

def canny_distance(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #img1 = cv2.equalizeHist(img1)
    #img2 = cv2.equalizeHist(img2)

    img1 = cv2.GaussianBlur(img1, (3, 3), 0)
    img2 = cv2.GaussianBlur(img2, (3, 3), 0)

    img1 = cv2.Canny(img1, 100, 120)
    img2 = cv2.Canny(img2, 100, 120)

    sp = img1.shape
    row = sp[0]
    column = sp[1]
    max = gcd(row, column)
    row = row//max
    column = column//max

    fd1, hist1 = hog(img1, orientations=9, pixels_per_cell=(row, column),
                        cells_per_block=(1, 1), feature_vector=False, visualize=True, channel_axis=False)
    fd2, hist2 = hog(img2, orientations=9, pixels_per_cell=(row, column),
                        cells_per_block=(1, 1), feature_vector=False, visualize=True, channel_axis=False)

    #fd1 = fd1.reshape(10, 10, -1)
    fd1 = fd1.squeeze()
    fd2 = fd2.squeeze()
    fd1 = np.array(fd1)
    fd2 = np.array(fd2)
    #print(hist1)
    #print('fd', fd2)
    #print("fd形状：", fd1.shape)  # (40, 40, 9)
    #print("hist形状：", hist1.shape)

    out = 0.0
    for i in range(max):
        for j in range(max):
            oud = 0
            for k in range(9):
                oud = oud + pow(fd1[i][j][k] - fd2[i][j][k], 2)
                #oud = oud + abs(fd1[i][j][k] - fd2[i][j][k])
            oud = np.sqrt(oud)
            out += oud
    out = out/(max*max)
    return out

if __name__ == '__main__':  # 测试边缘特征
    path1 = './static/images/lhy/001.jpg'
    path2 = './static/images/lhy/002.jpg'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    res = canny_distance(img1, img2)
    res = round(res, 4)
    print(res)



