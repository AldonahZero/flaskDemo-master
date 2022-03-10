import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import xlwt
from skimage import io
from sklearn.cluster import KMeans
import utils
import shutil
from PIL import ImageFont, ImageDraw, Image

'''
输入:
num: 原始图像对应的序号 此处为2 对应红外伪彩图像
path1: 输入原始图像的路径
path2: 输出的主色提取图像的存储路径
输出:
主色提取图像存储于path2
'''

def blob_kmeans(img, j):
    m = img.shape[0]
    n = img.shape[1]

    data = img.reshape((-1, 3))
    data = np.float32(data)

    # 定义终止条件 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 设置初始中心的选择
    # flags = cv2.KMEANS_RANDOM_CENTERS
    flags = cv2.KMEANS_PP_CENTERS

    # K-Means聚类 聚集成4类
    compactness, labels, centers = cv2.kmeans(data, j, None, criteria, 10, flags)

    dst = labels.reshape((img.shape[0], img.shape[1]))
    dst = np.uint8(dst)
    return dst


def hist_statistic(array):
    hist = [0 for i in range(101)]
    len_ = 0
    len_ = len(array)
    for i in range(len_):
        hist[int(array[i] * 100)] += 1
        print(int(array[i] * 100))

    return hist


def hist_square(hist_1, hist_2):
    length_of_hist = min(len(hist_1), len(hist_2))
    sum1 = 0
    sum2 = 0
    ex = 0
    ey = 0
    exy = 0
    dx = 0
    dy = 0
    cov = 0
    skewness = 0  # 目标图像的偏斜度，代表目标图像分布的位置
    # 协方差系数
    p = 0
    for i in range(length_of_hist):
        sum1 += hist_1[i]
        sum2 += hist_2[i]
    # 求灰度直方图的均值（期望）
    ex = sum1 / length_of_hist
    ey = sum2 / length_of_hist
    # 求平方的期望
    for i in range(length_of_hist):
        dx += pow(hist_1[i], 2)
        dy += pow(hist_2[i], 2)
    dx = dx / length_of_hist
    dy = dy / length_of_hist
    # 求方差=平方的期望减期望的平方
    dx = dx - pow(ex, 2)
    dy = dy - pow(ey, 2)
    # 求协方差cov=E[XY]-E[X]E[Y]
    for i in range(length_of_hist):
        exy += hist_1[i]*hist_2[i]
    exy = exy / length_of_hist
    cov = exy - ex*ey
    # 求协方差系数 p = cov / (dx开平方*dy开平方)
    p = cov / (pow(dx, 0.5)*pow(dy, 0.5))
    return ex, ey, exy, dx, dy, cov, p, skewness


def centroid_histogram(clt):

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((100, int(max(hist)*300), 3), dtype="uint8")
    # bar[np.where(bar==0)] = 255
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    percent_1 = []
    color_1 = []
    nums = 0
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = (percent * 300)
        cv2.rectangle(bar, (0, nums*20), (int(endX), (nums + 1)*20),
            color.astype("uint8").tolist(), -1)
        cv2.rectangle(bar, (int(endX), nums * 20), (300, (nums + 1) * 20),
                      [255, 255, 255], -1)
        # 绘制文字信息

        percent_1.append(percent)
        color_1.append(color)
        nums = nums + 1

    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, 15)
    img_pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, 0), "%.2f%%" % (hist[0] * 100), font=font, fill=(0, 0, 0))
    draw.text((0, 20), "%.2f%%" % (hist[1] * 100), font=font, fill=(0, 0, 0))
    draw.text((0, 40), "%.2f%%" % (hist[2] * 100), font=font, fill=(0, 0, 0))
    draw.text((0, 60), "%.2f%%" % (hist[3] * 100), font=font, fill=(0, 0, 0))
    draw.text((0, 80), "%.2f%%" % (hist[4] * 100), font=font, fill=(0, 0, 0))
    # return the bar chart
    bar = np.array(img_pil)
    return bar, percent_1, color_1


def c_main(img, k, num, path2):

    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img

    clt = KMeans(n_clusters=k)
    width = image.shape[0]
    length = image.shape[1]
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt.fit(image)

    labels = clt.labels_
    # print(labels)
    # print(clt.cluster_centers_)
    # print(len(labels))
    image_output = np.zeros((width, length, 3), dtype="uint8")

    for i in range(len(labels)):
        image_output[i//length, i%length] = clt.cluster_centers_[labels[i]]


    # image_output = image.reshape(width, length, 3)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(image_output[:, :, [2, 1, 0]])

    hist = utils.centroid_histogram(clt)
    bar, per, color = plot_colors(hist, clt.cluster_centers_)

    plt.subplot(1, 2, 2)
    plt.imshow(bar[:, :, [2, 1, 0]])
    # plt.savefig('static\\images_save\\main_color\\' + 'main_color' + str(num) + '.JPG')
    plt.savefig(path2 + 'main_color' + str(num) + '.JPG')
    plt.show()
    return bar, per, color


def mymain_color(img_input):
    if (len(os.listdir('static/images_save/main_color')) != 0):
        # 主色提取结果存储路径
        shutil.rmtree('static/images_save/main_color')
        os.mkdir('static/images_save/main_color')
    # img_input = cv2.imread('static\\images_GLCM_original\\images_camouflage\\mix\\20m\\' + str(num) +'.JPG')
    path2 = 'static/images_save/main_color/'
    c_main(img_input, k=5, num=2, path2=path2)
    return path2

# mymain_color(num=2)

