import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import xlwt
import math
#from skimage.feature import greycomatrix, greycoprops
#from skimage import io, color, img_as_ubyte


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def ou_distance1(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #img1 = cv2.equalizeHist(img1)
    #img2 = cv2.equalizeHist(img2)

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 255])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 255])
    #print(hist1.shape)

    '''match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    match3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    #print("巴氏距离：%s, 相关性：%s, 卡方：%s" % (match1, match2, match3))
    return match1'''

    m = 0
    n = 0
    m = len(img1.nonzero()[0])
    #print(m)
    n = len(img2.nonzero()[0])
    #print(n)
    for i in range(0, 256):
        hist1[i] = hist1[i] / m
        hist2[i] = hist2[i] / n
    oud = 0
    for i in range(0, 256):
        oud = oud + pow(hist1[i] - hist2[i], 2)
        #oud = oud + abs(hist1[i] - hist2[i])
    oud = np.sqrt(oud)
    return oud

'''list = []
for filename in os.listdir('D:\\work\\wz\\data\\beijing1\\original\\te2\\'):
    img = cv2.imread('D:\\work\\wz\\data\\beijing1\\original\\te2\\' + filename)
    #img = img[80:240, 160:460]
    img = img[110:270, 170:470]
    list.append(img)
for i in range(len(list)//2):
    match1 = ou_distance(list[2*i], list[2*i+1])
    print("巴氏距离：%s" % (match1))'''
    #out = ou_distance(list[2*i], list[2*i+1])
    #print(out)


# 定义表格
f = xlwt.Workbook(encoding = 'ascii')
sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
row0 = ["ou_distance of histogram"]
for i in range(0, len(row0)):
    sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

q = 1
p = 1
'''for filename in os.listdir('C:\\Users\\86155\\Desktop\\data\\wz_20_9\\'):  # 这一步是取文件中的文件名
    # 如果想同时取两个文件里面的对应图片，则需要两个文件里面对应图片相同
    # print(filename)
    if (q % 2 == 1):
        img1 = cv2.imread('C:\\Users\\86155\\Desktop\\data\\wz_20_9\\' + filename)

    elif(q % 2 == 0):
        img2 = cv2.imread('C:\\Users\\86155\\Desktop\\data\\wz_20_9\\' + filename)
        #sheet1.write_merge(p, p, 0, 0, "%.4f" % ou_distance(img1, img2))
        sheet1.write(p, 0, "%.4f" % ou_distance(img1, img2))
        print('欧式距离 = ', ou_distance(img1,img2))
        p = p + 1

    q = q + 1'''

'''img1 = cv2.imread('C:\\Users\\86155\\Desktop\\data\\rgbd\\pitcher\\pitcher80\\001.png')
sum = 0

for filename in os.listdir('C:\\Users\\86155\\Desktop\\data\\rgbd\\pitcher\\pitcher80\\')[1:]:  # 这一步是取文件中的文件名
    # 如果想同时取两个文件里面的对应图片，则需要两个文件里面对应图片相同
    print(filename)
    if (q % 2 == 0):
        img1 = cv2.imread('C:\\Users\\86155\\Desktop\\data\\rgbd\\pitcher\\pitcher80\\' + filename)

    elif(q % 2 == 1):
        img2 = cv2.imread('C:\\Users\\86155\\Desktop\\data\\rgbd\\pitcher\\pitcher80\\' + filename)
        #sheet1.write_merge(p, p, 0, 0, "%.4f" % ou_distance(img1, img2))
        sheet1.write(p, 0, "%.4f" % ou_distance(img1, img2))
        print('欧式距离 = ', ou_distance(img1, img2))
        sum = sum + ou_distance(img1, img2)
        average = sum / p
        print(p)
        print('平均值' "%.4f" % average)
        p = p + 1

    q = q + 1
#f.save('C:\\Users\\86155\\Desktop\\data\\rgbd\\cap.xls')'''

