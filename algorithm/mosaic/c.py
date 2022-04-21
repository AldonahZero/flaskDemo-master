import cv2
import numpy as np
import os
import re
import exifread
import xlwt

from common.file_tools import del_file

global img_t
global point1, point2
from PIL import Image
import screeninfo
import shutil
import glob
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import argparse
import math


MATCH_RESULT_PATH = 'algorithm/mosaic/static/matchresult/'
P2_PATH = 'algorithm/mosaic/static/P2/'

def on_mouse(event, x, y, flags, param):

    # 设置鼠标操作
    global img_t, point1, point2
    img2 = img_t.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('image', img2)

        # 得到左上点的坐标和所截区域的长宽，将所截区域保存
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img = img_t[min_y:min_y + height, min_x:min_x + width]
        cv2.imwrite('Selected_area.jpg', cut_img)


def overlap(box1, box2):

    # 判断两个矩形是否相交
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    area = (maxx - minx) * (maxy - miny)
    all_area = (maxx1 - minx1) * (maxy1 - miny1)
    o = area / all_area
    if minx > maxx or miny > maxy:
        return 0
    else:
        if o > 0.7:
            return 1


def distance(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2

    # 做差平方相加求两点距离
    lon = abs(lon1 - lon2)
    lat = abs(lat1 - lat2)
    return np.sum(lon**2 + lat**2)


def filehyperspectralGps(path_txts):
    num = 0
    for files in os.listdir(path_txts):
        file = os.path.splitext(path_txts + files)
        filename, type = file
        if type == '.dci' :
            num = num + 1

    x1 = np.zeros((num,2 ), dtype= np.double)
    num = 0

    for files in os.listdir(path_txts):
        # print(files)
        file1 = os.path.splitext(path_txts + files)
        filename, type = file1
        if type == '.dci':
            file = open(path_txts+"\\"+files , 'r')
            content = file.read()
            result1 = re.findall(".* latitude = (.*).*", content)
            result2 = re.findall(".* longitude = (.*).*", content)

            x1[num][0] = float(result1[0])
            x1[num][0] = 180 / math.pi * x1[num][0]

            x1[num][1] = (result2[0])
            x1[num][1] = 180 / math.pi * x1[num][1]

            num = num + 1
            file.close()
    return x1


def fileOptGps(path_img):

    # 读取可见光图片的GPS信息
    num = 0
    for files in os.listdir(path_img):
        num = num + 1
    x1 = np.zeros((num, 2), dtype=np.double)
    num = 0
    for files in os.listdir(path_img):

        # print(files)
        tags = exifread.process_file(open(path_img + '\\' + files, 'rb'))
        lat = tags["GPS GPSLatitude"].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
        lon = tags["GPS GPSLongitude"].printable[1:-1].replace(" ", "").replace("/", ",").split(",")

        # print('照片的经纬度：', (lat, lon), files)
        lat = float(lat[0]) + float(lat[1]) / 60 + float(lat[2]) / float(lat[3]) / 3600.0
        lon = float(lon[0]) + float(lon[1]) / 60 + float(lon[2]) / float(lon[3]) / 3600.0

        # print('照片的经纬度：',  (lat , lon) , files)
        x1[num][0] = float(lat)
        x1[num][1] = lon
        num = num + 1
    return x1


# def del_file(path_data):
#
#     # 删除文件夹下所有文件
#     for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
#         file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
#         if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
#             os.remove(file_data)
#         else:
#             del_file(file_data)


def get_file_name(path_date, string):

    # 提取文件夹下所有string类文件
    file_name = []
    for root, dirs, files in os.walk(path_date):
        for i in range(len(files)):
            if files[i][-3:] == string:
                file_name.append(files[i])
    return file_name


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def image_map(img_path, path_hw, path_kjg, path_ggp, key, pos):
    # global img_t

    # 路径
    path_result_ggp = MATCH_RESULT_PATH + 'result_ggp/' + key
    path_result_kjg = MATCH_RESULT_PATH + 'result_kjg/' + key
    path_result_hw = MATCH_RESULT_PATH + 'result_hw/' + key

    create_folder(path_result_ggp)
    create_folder(path_result_kjg)
    create_folder(path_result_hw)

    # 读取图片的长宽
    str_img = str(img_path)
    img1 = cv2.imread(str_img)

    x, y = img1.shape[0:2]

    # 删除result文件夹内的文件
    del_file(path_result_ggp)
    del_file(path_result_kjg)
    del_file(path_result_hw)

    # 可见光，红外和高光谱文件夹下的文件名
    file_hw = get_file_name(path_hw, 'JPG')
    file_kjg = get_file_name(path_kjg, 'JPG')
    file_ggp = []
    for root, dirs, files in os.walk(path_ggp):
        for i in range(len(files)):
            file_ggp.append(files[i])

    # 屏幕的长宽
    screen_id = 0
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    # 图片缩放的比例 ---> 输出图片
    # k = x / height + 0.2
    # img_t = cv2.resize(img1, (int(y / k), int(x / k)))
    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image', on_mouse)
    # cv2.imshow('image', img_t)
    # cv2.waitKey(0)


    # 所选区域的坐标
    box1 = np.array([int(pos[0][0]), int(pos[0][1]), int(pos[1][0]), int(pos[1][1])])

    # 取出P2.txt文件中的坐标数据
    file = open(P2_PATH + key + '.txt')
    lines = file.readlines()
    rows = len(lines)
    coordinates = np.zeros((rows, 5))
    row = 0
    for line in lines:
        line = line.strip().split(' ')
        coordinates[row, :] = line[:]
        row += 1

    # 比较所选区域和图片是否有交集
    for i in range(rows):
        box2 = np.array([coordinates[i, 1], coordinates[i, 2], coordinates[i, 3], coordinates[i, 4]])
        out = overlap(box1, box2)
        if out == 1:
            res = int(coordinates[i, 0]) + 1
            print(file_kjg[res])

            # 输出可见光和红外的图片
            # image_hw = Image.open('./红外/' + file_hw[res])
            # image_kjg = Image.open('./可见光/' + file_kjg[res])
            # # image_hw.show()
            # # image_kjg.show()
            # plt.imshow(image_kjg)
            # plt.show()

            # 将可见光的图片放到result_kjg文件夹内
            res_kjg = path_kjg + '/' + file_kjg[res]
            result_kjg = path_result_kjg + '/' + file_kjg[res]
            shutil.copy(res_kjg, result_kjg)

            # 将红外的图片放到result_hw文件夹内
            res_hw = path_hw + '/' + file_hw[res]
            result_hw = path_result_hw + '/' + file_hw[res]
            shutil.copy(res_hw, result_hw)

            # 高光谱和可见光图片的GPS信息
            gps_kjg = fileOptGps(path_result_kjg)
            gps_ggp = filehyperspectralGps(path_ggp)

            # 比较GPS信息，将高光谱文件放入result文件夹
            x1 = []
            for j in range(len(gps_kjg)):
                for k in range(len(gps_ggp)):
                    dis = distance(gps_kjg[j][0], gps_kjg[j][1], gps_ggp[k][0], gps_ggp[k][1]) * 1e+6
                    x1.append(dis)
                x2 = np.array(x1)
                x3 = x2.argsort()
                x4 = x3.argsort()
                for x in range(len(x4)):
                    if (x4[x] == 0) or (x4[x] == 1):
                        for ggp in range(5):
                            res_ggp = path_ggp + '/' + file_ggp[5*x+ggp]
                            result_ggp = path_result_ggp + '/' + file_ggp[5*x+ggp]
                            if not os.path.exists(result_ggp):
                                shutil.copy(res_ggp, result_ggp)
                x1 = []

           # cv2.destroyAllWindows()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--path_hw', type=str, default='./红外')
#     parser.add_argument('--path_kjg', type=str, default='./可见光')
#     parser.add_argument('--path_ggp', type=str, default='./300米牛耕文件')
#     parser.add_argument('--image', type=str, default='mosRes_2022_03_22_10_47_13_366.png')
#
#     args = parser.parse_args()
#
#     image_map(args)