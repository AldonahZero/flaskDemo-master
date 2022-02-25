import os
import cv2
import numpy as np
import pandas as pd
from canny_hog import ou_distance2
from histogram_ou_distance import ou_distance1

def grey_compare(pics):  #灰度特征
    list1 = []
    list2 = []
    sum = 0.0
    for pic in pics:
        img = cv2.imread(pic.url)
        list1.append(img)

    list1 = list1 + list1
    for interval in range(30):
        for i in range(len(list1) // 2):
            dis = ou_distance1(list1[i], list1[i + interval])
            sum = sum + dis
        average = sum / (len(list1) // 2)
        list2.append(np.float(average[0]))

    return list2


def canny_compare(path):  #边缘特征
    list1 = []
    list2 = []
    sum = 0.0
    for filename in os.listdir(path):
        img = cv2.imread(path + filename)
        list1.append(img)

    list1 = list1 + list1
    for interval in range(30):
        for i in range(len(list1) // 2):
            dis = ou_distance2(list1[i], list1[i + interval])
            sum = sum + dis
        average = sum / (len(list1) // 2)
        list2.append(average)

    list2 = [str(i) for i in list2]  # 列表转为字符串
    string = ' '.join(list2)
    print('string', string)

    return string