import cv2
import numpy as np
from algorithm.multiplePerspectives.canny_hog import canny_distance
from algorithm.multiplePerspectives.histogram_ou_distance import grey_distance
from algorithm.multiplePerspectives.lbp import lbp_distance
from algorithm.multiplePerspectives.kaze import kaze_distance

def grey_compare(pics):  #灰度特征
    list1 = []
    list2 = []
    for pic in pics:
        img = cv2.imread(pic.url)
        list1.append(img)

    list1 = list1 + list1
    for interval in range(16):
        sum = 0.0
        for i in range(len(list1) // 2):
            dis = grey_distance(list1[i], list1[i + interval])
            sum = sum + dis
        average = sum / (len(list1) // 2)
        average = np.round(average, 4)
        list2.append(np.float(average[0]))
    return list2  # 返回一个浮点数列表

def canny_compare(pics):  #边缘特征
    list1 = []
    list2 = []
    for pic in pics:
        print()
        img = cv2.imread(pic.url)
        list1.append(img)

    list1 = list1 + list1
    for interval in range(16):
        sum = 0.0
        for i in range(len(list1) // 2):
            dis = canny_distance(list1[i], list1[i + interval])
            sum = sum + dis
        average = sum / (len(list1) // 2)
        average = np.round(average, 4)
        list2.append(np.float(average))
    return list2  # 返回一个浮点数列表

def lbp_compare(pics):  #lbp纹理特征
    list1 = []
    list2 = []
    for pic in pics:
        img = cv2.imread(pic.url)
        list1.append(img)

    list1 = list1 + list1
    for interval in range(16):
        sum = 0.0
        for i in range(len(list1) // 2):
            dis = lbp_distance(list1[i], list1[i + interval])
            sum = sum + dis
        average = sum / (len(list1) // 2)
        average = np.round(average, 4)
        list2.append(np.float(average))
    return list2  # 返回一个浮点数列表

def kaze_compare(pics):  #kaze角点特征
    list1 = []
    list2 = []
    for pic in pics:
        img = cv2.imread(pic.url)
        list1.append(img)

    list1 = list1 + list1
    for interval in range(16):
        sum = 0.0
        for i in range(len(list1) // 2):
            dis = kaze_distance(list1[i], list1[i + interval])
            sum = sum + dis
        average = sum / (len(list1) // 2)
        average = np.round(average, 1)
        list2.append(np.float(average   ))
    return list2  # 返回一个浮点数列表