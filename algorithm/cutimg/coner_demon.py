import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import xlwt
import os


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def sift_kp(image):
    # sift = cv2.xfeatures2d_SIFT.create()
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 0, 255))
    return kp_image, kp, des


def surf_kp(image):
    surf = cv2.xfeatures2d_SURF.create()
    kp, des = surf.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 0, 255))
    return kp_image, kp, des


def myconer(path, path_save_coner):
    # path = 'static\\images_GLCM'
    # path_save_coner = 'static/images_save/coner/'

    for filename1 in os.listdir(path):
        if not filename1.startswith('.'):
            path1 = path + '/' + filename1

            for filename2 in os.listdir(path1):
                if not filename2.startswith('.'):
                    path2 = path1 + '/' + filename2
                    for filename3 in os.listdir(path2):
                        if not filename3.startswith('.'):
                            path3 = path2 + '/' + filename3
                            # print(path3)
                        # q = q + 1

                            for w in range(10):  # 十个角度
                                sift = cv2.xfeatures2d.SIFT_create()

                                target_name = str(w) + str(4) + '.JPG'

                                for j in range(9):  # 周围八个背景+目标
                                    # print('j= ', j)
                                    if j == 4:
                                        continue
                                    backg_name = str(w) + str(j) + '.JPG'
                                    # print(backg_name)
                                    if not os.path.exists(path3 + '/' + backg_name):
                                        continue
                                    img_backg = cv2.imread(path3 + '/' + backg_name)
                                    if img_backg is None:
                                        continue

                                    if img_backg.ndim == 3:
                                        back_gray = cv2.cvtColor(img_backg, cv2.COLOR_BGR2GRAY)
                                    elif img_backg.ndim == 1:
                                        back_gray = img_backg
                                    else:
                                        print('Please input the correct image!')

                                    back_image, _, back_des = sift_kp(back_gray)

                                    img_target = cv2.imread(path3 + '/' + target_name)
                                    if img_target is None:
                                        continue
                                    if img_target.ndim == 3:
                                        target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
                                    elif img_target.ndim == 1:
                                        target_gray = img_target
                                    else:
                                        print('Please input the correct image!')

                                    kp1, des1 = sift.detectAndCompute(back_gray, None)
                                    kp2, des2 = sift.detectAndCompute(target_gray, None)
                                    bf = cv2.BFMatcher()
                                    matches = bf.knnMatch(des1, des2, k=2)

                                    matchesMask = [[0, 0] for i in range(len(matches))]
                                    Max = 0
                                    Min = 999
                                    count = 0
                                    for i, (m1, m2) in enumerate(matches):
                                        if m1.distance < 0.85 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
                                            matchesMask[i] = [1, 0]
                                            count = count + 1
                                            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
                                            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
                                            # print(kpts1)
                                            # print(i, pt1, pt2)
                                            a = ((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[0] - pt2[0]) * (
                                                    pt1[0] - pt2[0])) ** 0.5

                                            if a >= Max:
                                                Max = a
                                            if a < Min:
                                                Min = a

                                            if i % 5 == 0:
                                                cv2.circle(back_gray, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 255), -1)
                                                cv2.circle(target_gray, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 255),
                                                           -1)

                                    # 匹配点为蓝点, 坏点为红点
                                    draw_params = dict(matchColor=(255, 0, 0),
                                                       singlePointColor=(0, 0, 255),
                                                       matchesMask=matchesMask,
                                                       flags=0)

                                    res = cv2.drawMatchesKnn(back_gray, kp1, target_gray, kp2, matches, None,
                                                             **draw_params)

                                    print(j)

                                    plt.subplot(3, 3, j + 1)
                                    plt.title('%.2f%%' % (100 * count / len(matches)))

                                    print(count)

                                    print(len(matches))
                                    plt.imshow(back_image[:, :, [2, 1, 0]])

                                img_target2 = cv2.imread(path3 + '/' + target_name)
                                if img_target2 is None:
                                    continue
                                if img_target2.ndim == 3:
                                    target_gray2 = cv2.cvtColor(img_target2, cv2.COLOR_BGR2GRAY)
                                elif img_target2.ndim == 1:
                                    target_gray2 = img_target2
                                else:
                                    print('Please input the correct image!')

                                target_image2, _, target_des2 = sift_kp(target_gray2)

                                plt.subplot(3, 3, 5)
                                plt.title('target')
                                plt.imshow(target_image2[:, :, [2, 1, 0]])
                            '''
                            plt.savefig('static\\images_save\\coner\\coner.JPG')
                            '''
                            plt.savefig(path_save_coner + '/coner.JPG')
                            # plt.show()

    return path_save_coner
# myconer()
