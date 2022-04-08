
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


# GLCM properties


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
    # print("============BRISKF=================")
    return BRISK_count, BRISK_len_count


def ORB(img1, img2):
    orb = cv2.ORB_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
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
            count = count + 1
            #print(i, pt1, pt2)
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

    # print("============ORB=================")
    ORB_count = count
    ORB_len_count = len(matches)
    return ORB_count, ORB_len_count


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

    # print("============SIFT=================")
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

    # print("============SURF=================")
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

    # print("============FAST=================")
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

    # print("============KAZE=================")
    KAZE_count = count
    # if (len(matches) == None) :
    #     KAZE_len_count = 0
    # else:
    #     KAZE_len_count = len(matches)
    #
    KAZE_len_count = len(matches)
    return KAZE_count, KAZE_len_count


def myConer_excelSave(path_cutimg, excels_coner_coner):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('BRISK', cell_overwrite_ok=True)

    row0 = ["角点特征", "BRISKF", "ORB", "SIFT", "SURF", "FAST", "KAZE"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径
    # excels_coner_coner = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/coner_coner/'  # coner表格存储路径


    target_name = str(1) + str(4) + '.jpg'
    img_target = cv2.imread(path_cutimg + target_name)

    # 计算目标
    gray_target = img_target
    gray_target = img_as_ubyte(gray_target)

    [a1, b1, c1] = img_target.shape
    BRISK_a = 0
    BRISK_b = 0

    ORB_a = 0
    ORB_b = 0

    SIFT_a = 0
    SIFT_b = 0

    SURF_a = 0
    SURF_b = 0

    FAST_a = 0
    FAST_b = 0

    KAZE_a = 0
    KAZE_b = 0

    cnt_brisk = 0
    cnt_sift = 0
    cnt_surf = 0
    cnt_orb = 0
    cnt_kaze = 0
    cnt_fast = 0

    sign = True
    sign_error = True
    for j in range(9):
        # print(j)
        if j == 4:
            continue
        backg_name = str(1) + str(j) + '.jpg'
        if not os.path.exists(path_cutimg + backg_name):
            continue
        img_backg = cv2.imread(path_cutimg + backg_name)
        if img_backg is None:
            continue
        [a2, b2, c2] = img_backg.shape

        # 修改函数#########################################################################
        gray_bg = img_backg
        gray_bg = img_as_ubyte(gray_bg)

        try:
            a1, b1 = BRISK(gray_target, gray_bg)
            BRISK_a = BRISK_a + a1
            BRISK_b = BRISK_b + b1
            cnt_brisk = cnt_brisk + 1
        except:
            BRISK_a = BRISK_a
            BRISK_b = BRISK_b
            pass

        try:
            a_orb, b_orb = ORB(gray_target, gray_bg)
            ORB_a = ORB_a + a_orb
            ORB_b = ORB_b + b_orb
            cnt_orb = cnt_orb + 1
        except:
            ORB_a = ORB_a
            ORB_b = ORB_b

            pass

        try:
            a3, b3 = SIFT(gray_target, gray_bg)
            SIFT_a = SIFT_a + a3
            SIFT_b = SIFT_b + b3
            cnt_sift = cnt_sift + 1
        except:
            SIFT_a = SIFT_a
            SIFT_b = SIFT_b

            pass

        try:
            a4, b4 = SURF(gray_target, gray_bg)
            SURF_a = SURF_a + a4
            SURF_b = SURF_b + b4
            cnt_surf = cnt_surf + 1
        except:
            SURF_a = SURF_a
            SURF_b = SURF_b

            pass

        try:
            a5, b5 = FAST(gray_target, gray_bg)
            FAST_a = FAST_a + a5
            FAST_b = FAST_b + b5
            cnt_fast = cnt_fast + 1
        except:
            FAST_a = FAST_a
            FAST_b = FAST_b

            pass

        try:
            a6, b6 = KAZE(gray_target, gray_bg)
            KAZE_a = KAZE_a + a6
            KAZE_b = KAZE_b + b6
            cnt_kaze = cnt_kaze + 1
        except:
            KAZE_a = KAZE_a
            KAZE_b = KAZE_b

            pass

        # 以上为修改函数####################################################################
        # print(cnt)
        sign = False
    if sign:
        everage_BRISK_a = 'NA'
        everage_BRISK_b = 'NA'

        everage_ORB_a = 'NA'
        everage_ORB_b = 'NA'

        everage_SIFT_a = 'NA'
        everage_SIFT_b = 'NA'

        everage_SURF_a = 'NA'
        everage_SURF_b = 'NA'

        everage_FAST_a = 'NA'
        everage_FAST_b = 'NA'

        everage_KAZE_a = 'NA'
        everage_KAZE_b = 'NA'


    else:
        if cnt_brisk != 0:
            everage_BRISK_a = BRISK_a / cnt_brisk
            everage_BRISK_b = BRISK_b / cnt_brisk
        else:
            everage_BRISK_a = 123456789
            everage_BRISK_b = 123456789

        if cnt_orb != 0:
            everage_ORB_a = ORB_a / cnt_orb
            everage_ORB_b = ORB_b / cnt_orb
        else:
            everage_ORB_a = 123456789
            everage_ORB_b = 123456789

        if cnt_sift != 0:
            everage_SIFT_a = SIFT_a / cnt_sift
            everage_SIFT_b = SIFT_b / cnt_sift
        else:
            everage_SIFT_a = 123456789
            everage_SIFT_b = 123456789
        if cnt_surf != 0:
            everage_SURF_a = SURF_a / cnt_surf
            everage_SURF_b = SURF_b / cnt_surf
        else:
            everage_SURF_a = 123456789
            everage_SURF_b = 123456789

        if cnt_fast != 0:
            everage_FAST_a = FAST_a / cnt_fast
            everage_FAST_b = FAST_b / cnt_fast
        else:
            everage_FAST_a = 123456789
            everage_FAST_b = 123456789

        if cnt_kaze != 0:
            everage_KAZE_a = KAZE_a / cnt_kaze
            everage_KAZE_b = KAZE_b / cnt_kaze
        else:
            everage_KAZE_a = 123456789
            everage_KAZE_b = 123456789


    sheet1.write_merge(1, 1, 0, 0, "匹配成功角点数")

    sheet1.write_merge(1, 1, 1, 1, everage_BRISK_a)
    sheet1.write_merge(1, 1, 2, 2, everage_ORB_a)
    sheet1.write_merge(1, 1, 3, 3, everage_SIFT_a)
    sheet1.write_merge(1, 1, 4, 4, everage_SURF_a)
    sheet1.write_merge(1, 1, 5, 5, everage_FAST_a)
    sheet1.write_merge(1, 1, 6, 6, everage_KAZE_a)

    sheet1.write_merge(2, 2, 0, 0, "检测角点总数")

    sheet1.write_merge(2, 2, 1, 1, everage_BRISK_b)
    sheet1.write_merge(2, 2, 2, 2, everage_ORB_b)
    sheet1.write_merge(2, 2, 3, 3, everage_SIFT_b)
    sheet1.write_merge(2, 2, 4, 4, everage_SURF_b)
    sheet1.write_merge(2, 2, 5, 5, everage_FAST_b)
    sheet1.write_merge(2, 2, 6, 6, everage_KAZE_b)

    f.save(excels_coner_coner + 'excel_coner_coner.xls')

    return excels_coner_coner

# myConer_excelSave()