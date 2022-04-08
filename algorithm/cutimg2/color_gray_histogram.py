import os
import cv2
import xlwt
import matplotlib.pyplot as plt
import math
import numpy as np

def myGrayHitogram(path_original, path_bitwise):

    q = 0

    # path_original = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_original/1.jpg'
    # path_bitwise = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_bitwise/1.jpg'

    array_t_length = 0
    array_b_length = 0

    bitwise = cv2.imread(path_bitwise, 0)

    gray_original = cv2.imread(path_original, 0)

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

    t = 0
    b = 0
    A = gray_original

    for i in range(height - 1):
        for j in range(width - 1):
            pixel_value = A[i, j]
            pixel_value2 = bitwise[i, j]
            if pixel_value2 != 0:
                array_t[t] = pixel_value
                t = t + 1

    array_t = np.array(array_t, np.float32)

    hist_t = cv2.calcHist([array_t], [0], None, [256], [0, 255])
    arr = np.zeros(256)
    for i in range(256):
        arr[i] = hist_t[i][0]

    return arr

# print(myGrayHitogram())


