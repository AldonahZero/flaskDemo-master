import os
import cv2
import xlwt
import matplotlib.pyplot as plt
import math
import numpy as np

def myGrayMean(path_cutimg):
    q = 10

    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径

    arr = np.zeros(9)

    for filename1 in os.listdir(path_cutimg):
        if not os.path.exists(path_cutimg + '/' + filename1):
            q += 1
            continue
        # print(filename1)
        img = cv2.imread(path_cutimg + '/' + filename1, 0);
        height, width = img.shape
        size = img.size

        average = 0
        for i in range(height):
            for j in range(width):
                average += img[i][j] / size
        arr[q - 10] = average
        q += 1
    return arr

# print(myGrayMean())