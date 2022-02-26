import cv2
import numpy as np
import pandas as pd

def origin_LBP(img):
    dst = np.zeros(img.shape, dtype=img.dtype)
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            #print(j)
            center = img[i][j]
            code = 0

            code |= (img[i - 1][j - 1] >= center) << (np.uint8)(7)
            code |= (img[i - 1][j] >= center) << (np.uint8)(6)
            code |= (img[i - 1][j + 1] >= center) << (np.uint8)(5)
            code |= (img[i][j + 1] >= center) << (np.uint8)(4)
            code |= (img[i + 1][j + 1] >= center) << (np.uint8)(3)
            code |= (img[i + 1][j] >= center) << (np.uint8)(2)
            code |= (img[i + 1][j - 1] >= center) << (np.uint8)(1)
            code |= (img[i][j - 1] >= center) << (np.uint8)(0)

            dst[i - 1][j - 1] = code
    return dst

def getLBPH(img_lbp, numPatterns, grid_x, grid_y):
    '''
    计算LBP特征图像的直方图LBPH
    '''
    h,w=img_lbp.shape
    width = int(w / grid_x)
    height = int(h / grid_y)
    # 定义LBPH的行和列，grid_x*grid_y表示将图像分割的块数，numPatterns表示LBP值的模式种类
    result = np.zeros((grid_x * grid_y, numPatterns), dtype=float)
    resultRowIndex = 0
    # 对图像进行分割，分割成grid_x*grid_y块，grid_x，grid_y默认为8
    for i in range(grid_x):
        for j in range(grid_y):
            # 图像分块
            src_cell = img_lbp[i*height:(i+1)*height, j*width:(j+1)*width]
            # 计算直方图
            hist_cell = getLocalRegionLBPH(src_cell, 0, (numPatterns-1), True)
            # 将直方图放到result中
            result[resultRowIndex] = hist_cell
            resultRowIndex += 1
    #print(result.shape)
    return np.reshape(result, (-1))

def getLocalRegionLBPH(src, minValue, maxValue, normed):
    '''
    计算一个LBP特征图像块的直方图
    '''
    data = np.reshape(src, (-1))
    # 计算得到直方图bin的数目，直方图数组的大小
    bins = maxValue - minValue + 1
    # 定义直方图每一维的bin的变化范围
    ranges = (float(minValue), float(maxValue + 1))
    # hist, bin_edges = np.histogram(src, bins=bins, range=ranges, density=density)
    #hist, bin_edges = np.histogram(src, bins=bins, range=ranges, normed=normed)
    hist, bin_edges = np.histogram(src, bins=bins, range=ranges)
    # normed = normed
    return hist

def lbp_distance(img1, img2):  # 计算两幅图lbp特征距离
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    uniform_pattern1 = origin_LBP(img1)
    uniform_pattern2 = origin_LBP(img2)
    lbph1 = getLBPH(uniform_pattern1, 59, 8, 8)
    lbph2 = getLBPH(uniform_pattern2, 59, 8, 8)
    series1 = pd.Series(lbph1)
    series2 = pd.Series(lbph2)
    corr = round(series1.corr(series2, method='kendall'), 4)
    return corr

if __name__ == '__main__':  # 测试lbp特征
    path1 = './static/images/lhy/001.jpg'
    path2 = './static/images/lhy/002.jpg'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    res = lbp_distance(img1, img2)
    res = round(res, 4)
    print(res)