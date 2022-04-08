
import numpy as np
import cv2
import os
import xlwt
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

# 设置表格样式


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def BRISK(img1, img2):

    brisk = cv2.BRISK_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = brisk.detect(img1, None)
    kp2 = brisk.detect(img2, None)
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ## Ratio test
    # print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    Max = 0
    Min = 999
    count = 0
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.85 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            # print(i, pt1, pt2)
            count = count+1
            a = ((pt1[0]-pt2[0])*(pt1[0]-pt2[0])+(pt1[0]-pt2[0])*(pt1[0]-pt2[0]))**0.5

            if a > Max:
                Max = a
            if a < Min:
                Min = a

            if i % 5 == 0:
                cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 255), -1)
                cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 255), -1)

    # 匹配点为蓝点, 坏点为红点
    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=0)

    BRISK_count = count
    BRISK_len_count = len(matches)
    print("============BRISKF=================")
    return BRISK_count, BRISK_len_count


def ORB(img1):
    orb = cv2.ORB_create()

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray, None)

    cv2.drawKeypoints(gray, kp1, img1)

    cv2.imshow('orb', img1)
    cv2.waitKey(0)
    return


def SIFT(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ## Ratio test
    #print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    Max = 0
    Min = 999
    count = 0
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.85 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            count = count+1
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            #print(i, pt1, pt2)
            a = ((pt1[0]-pt2[0])*(pt1[0]-pt2[0])+(pt1[0]-pt2[0])*(pt1[0]-pt2[0]))**0.5

            if a > Max:
                Max = a
            if a < Min:
                Min = a

            if i % 5 == 0:
                cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 255), -1)
                cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 255), -1)

    # 匹配点为蓝点, 坏点为红点
    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=0)

    print("============SIFT=================")
    SIFT_count = count
    SIFT_len_count = len(matches)
    return SIFT_count, SIFT_len_count


def SURF(img1, img2):

    surf = cv2.xfeatures2d.SURF_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ## Ratio test
    #print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    Max = 0
    Min = 999
    count = 0
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.85 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            count = count+1
            #(i, pt1, pt2)
            a = ((pt1[0]-pt2[0])*(pt1[0]-pt2[0])+(pt1[0]-pt2[0])*(pt1[0]-pt2[0]))**0.5

            if a > Max:
                Max = a
            if a < Min:
                Min = a

            if i % 5 == 0:
                cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 255), -1)
                cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 255), -1)

    # 匹配点为蓝点, 坏点为红点
    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=0)

    print("============SURF=================")
    SURF_count = count
    SURF_len_count = len(matches)
    return SURF_count, SURF_len_count


def FAST(img1, img2):

    fast = cv2.FastFeatureDetector_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ## Ratio test
    #print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    Max = 0
    Min = 999
    count = 0
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.85 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            #(i, pt1, pt2)
            count = count+1
            a = ((pt1[0]-pt2[0])*(pt1[0]-pt2[0])+(pt1[0]-pt2[0])*(pt1[0]-pt2[0]))**0.5

            if a > Max:
                Max = a
            if a < Min:
                Min = a

            if i % 5 == 0:
                cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 255), -1)
                cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 255), -1)

    # 匹配点为蓝点, 坏点为红点
    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=0)

    print("============FAST=================")
    FAST_count = count
    FAST_len_count = len(matches)
    return FAST_count, FAST_len_count


def KAZE(img1, img2):
    fast = cv2.KAZE_create()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ## Ratio test
    #print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    Max = 0
    Min = 999
    count = 0
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.85 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            #print(i, pt1, pt2)
            count = count + 1
            a = ((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[0] - pt2[0]) * (pt1[0] - pt2[0])) ** 0.5

            if a > Max:
                Max = a
            if a < Min:
                Min = a

            if i % 5 == 0:
                cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 255), -1)
                cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 255), -1)

    # 匹配点为蓝点, 坏点为红点
    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=0)

    print("============KAZE=================")
    KAZE_count = count
    # if (len(matches) == None) :
    #     KAZE_len_count = 0
    # else:
    #     KAZE_len_count = len(matches)
    #
    KAZE_len_count = len(matches)
    return KAZE_count, KAZE_len_count



def myConer(path_cutimg, path_coner, path_coner_ORB, path_coner_FAST, path_coner_SURF, path_coner_SIFT,
            path_coner_BRISKF, path_coner_KAZE):
    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径
    # path_coner = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/'
    # path_coner_ORB = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/ORB/'
    # path_coner_FAST = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/FAST/'
    # path_coner_SURF = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/SURF/'
    # path_coner_SIFT = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/SIFT/'
    # path_coner_BRISKF = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/BRISKF/'
    # path_coner_KAZE = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/KAZE/'

    path = path_cutimg + '14.jpg'

    # -------------------------------------------ORB---------------------------------------------------
    img1 = cv2.imread(path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    cv2.drawKeypoints(gray1, kp1, img1)
    cv2.imwrite(path_coner_ORB + 'ORB.jpg', img1)

    # -------------------------------------------SIFT---------------------------------------------------
    img2 = cv2.imread(path)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp2, dst2 = sift.detectAndCompute(gray2, None)  # 第二个参数为mask区域
    cv2.drawKeypoints(gray2, kp2, img2)
    cv2.imwrite(path_coner_SIFT + 'SIFT.jpg', img2)

    # -------------------------------------------SURF---------------------------------------------------
    img3 = cv2.imread(path)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create()
    kp3, dst3 = surf.detectAndCompute(gray3, None)  # 第二个参数为mask区域
    cv2.drawKeypoints(gray3, kp3, img3)
    cv2.imwrite(path_coner_SURF + 'SURF.jpg', img3)

    # -------------------------------------------KAZE---------------------------------------------------
    img4 = cv2.imread(path)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

    kaze = cv2.KAZE_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp4 = kaze.detect(gray4, None)
    kp4, dst4 = brief.compute(gray4, kp4)
    cv2.drawKeypoints(gray4, kp4, img4)
    cv2.imwrite(path_coner_KAZE + 'KAZE.jpg', img4)

    # -------------------------------------------FAST---------------------------------------------------
    img5 = cv2.imread(path)
    gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp5 = fast.detect(gray5, None)
    kp5, des5 = brief.compute(gray5, kp5)
    cv2.drawKeypoints(gray5, kp5, img5)
    cv2.imwrite(path_coner_FAST + 'FAST.jpg', img5)

    # -------------------------------------------BRISKF-------------------------------------------------
    img6 = cv2.imread(path)
    gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

    brisk = cv2.BRISK_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp6 = brisk.detect(gray6, None)
    kp6, des6 = brief.compute(gray6, kp6)
    cv2.drawKeypoints(gray6, kp6, img6)
    cv2.imwrite(path_coner_BRISKF + 'BRISKF.jpg', img6)

    return path_coner

# myConer()


