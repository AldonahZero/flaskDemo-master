import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import xlwt


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


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


def myBlob_excelSave(path_cutimg, excels_save_blob_Kmeans, path_blob_hist_save):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('Blob', cell_overwrite_ok=True)

    row0 = ["斑块统计特征-目标背景相似度", "面积", "周长", "圆形度", "矩形度", "伸长度"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    array_area1 = []
    array_length1 = []
    array_circle1 = []
    array_rect1 = []
    array_stretch1 = []

    array_area2 = []
    array_length2 = []
    array_circle2 = []
    array_rect2 = []
    array_stretch2 = []

    # path = '18'  # 10-18
    # path = str(21)

    k = 5
    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径
    # excels_blob_Kmeans = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/blob_Kmeans/'  # Tamura表格存储路径
    # path_blob_hist_save = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_blob/' #生成的斑块图像与统计直方图存储路径

    path1 = path_cutimg

    img_input = cv2.imread(os.path.join(path1,'14.jpg'))
    k2 = np.ones((3, 3), np.uint8)  # 开运算算子
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
    img_input = cv2.medianBlur(img_input, 5)

    img = cv2.cvtColor(img_input, cv2.COLOR_BGR2LAB)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j][0] = 0
    img1 = img

    # plt.figure("Image_0ab")
    # plt.imshow(img)

    # k = 7
    dst1 = blob_kmeans(img1, k)
    dst2 = np.zeros((img1.shape), np.uint8)
    color_type = [[0, 0, 0],
                  [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255],
                  [255, 0, 255], [0, 255, 255],
                  [0, 0, 128], [0, 128, 0], [128, 0, 0], [128, 128, 0], [128, 128, 128],
                  [128, 0, 128], [0, 128, 128],
                  [0, 0, 64], [0, 64, 0], [64, 0, 0], [64, 64, 0], [64, 64, 64], [64, 0, 64], [0, 64, 64],
                  [0, 0, 192], [0, 192, 0], [192, 0, 0], [192, 192, 0], [192, 192, 192], [192, 0, 192],
                  [0, 192, 192]
                  ]

    dst3 = np.zeros((img1.shape[0], img1.shape[1], k), np.uint8)

    for i in range(k):
        point = np.argwhere(dst1 == i)
        for a, b in point:
            dst2[a, b] = color_type[i]
            dst3[a, b, i] = 255

    # cv2.imshow("result", dst2)
    # cv2.imshow("dst3", dst3[:, :, 1])
    # cv2.imshow("dst2", dst2)
    # cv2.imwrite('static\\images_save\\blob_hist\\' + 'blob' + '14.jpg', dst2)
    cv2.imwrite(os.path.join(path_blob_hist_save, 'blob' + '14.jpg'), dst2)

    area_ = []
    length_ = []
    # compact_ = []
    circle_ = []
    rect_ = []
    stretch_ = []  # 周长除以面积

    for n in range(k):
        dst4 = cv2.morphologyEx(dst3[:, :, n], cv2.MORPH_OPEN, k2)  # 开运算 去除细小白色区域
        # cv2.imshow("dst4", dst4)

        dst5 = cv2.morphologyEx(dst4, cv2.MORPH_CLOSE, k2)  # 开运算 去除细小白色区域
        # cv2.imshow("dst5", dst5)

        dst6 = cv2.merge((dst5, dst5, dst5))
        image, contours, hierarchy = cv2.findContours(dst5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # print('length of contours = ', len(contours))
        num_contours = len(contours)

        for i in range(num_contours):
            area_i = cv2.contourArea(contours[i])
            area_.append(area_i)

            length_i = cv2.arcLength(contours[i], True)
            length_.append(length_i)

            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print("box = ", box)
            if box[0][0] == box[2][0] or box[0][1] == box[2][1]:
                area_rect_i = 1
            else:
                area_rect_i = abs(box[0][0] - box[2][0]) * abs(box[0][1] - box[2][1])
                rect_i = area_i / area_rect_i
            rect_.append(rect_i)

            circle_i = 4 * np.pi * area_i / length_i / length_i
            circle_.append(circle_i)

            stretch_i = length_i / area_i
            stretch_.append(stretch_i)

    area_ = np.array(area_, np.float32)
    length_ = np.array(length_, np.float32)
    rect_ = np.array(rect_, np.float32)
    circle_ = np.array(circle_, np.float32)
    stretch_ = np.array(stretch_, np.float32)

    array_area1 = area_
    array_length1 = length_
    array_circle1 = circle_
    array_rect1 = rect_
    # print('array_rect1 = ', array_rect1)
    array_stretch1 = stretch_

    # array_area1 = array_area1/max(array_area1)
    # array_length1 = array_length1 / max(array_length1)
    # array_circle1 = array_circle1
    # array_rect1 = array_rect1
    # array_stretch1 = array_stretch1
    array_area1 = np.trunc(100 * array_area1 / max(array_area1)) / 100
    array_length1 = np.trunc(100 * array_length1 / max(array_length1)) / 100
    array_circle1 = np.trunc(100 * array_circle1 / max(array_circle1)) / 100
    array_rect1 = np.trunc(100 * array_rect1 / max(array_rect1)) / 100
    # print('array_rect1 = ', array_rect1)
    array_stretch1 = np.trunc(100 * array_stretch1 / max(array_stretch1)) / 100

    plt.figure('area_14')
    n, bins, patches = plt.hist(array_area1, 100, (0, 1))
    # plt.savefig('static\\images_save\\blob_hist\\' + 'area_' + '14.jpg')
    plt.savefig(path_blob_hist_save +os.path.sep+ 'area_' + '14.jpg')
    plt.close()

    plt.figure('length_14')
    n, bins, patches = plt.hist(array_length1, 100, (0, 1))
    # plt.savefig('static\\images_save\\blob_hist\\' + 'length_' + '14.jpg')
    plt.savefig(path_blob_hist_save +os.path.sep+ 'length_' + '14.jpg')
    plt.close()

    plt.figure('rect_14')
    n, bins, patches = plt.hist(array_rect1, 100, (0, 1))
    # plt.savefig('static\\images_save\\blob_hist\\' + 'rect_' + '14.jpg')
    plt.savefig(path_blob_hist_save +os.path.sep+ 'rect_' + '14.jpg')
    plt.close()

    plt.figure('circle_14')
    n, bins, patches = plt.hist(array_circle1, 100, (0, 1))
    # plt.savefig('static\\images_save\\blob_hist\\' + 'circle_' + '14.jpg')
    plt.savefig(path_blob_hist_save +os.path.sep+ 'circle_' + '14.jpg')
    plt.close()

    plt.figure('stretch_14')
    n, bins, patches = plt.hist(array_stretch1, 100, (0, 1))
    # plt.savefig('static\\images_save\\blob_hist\\' + 'stretch_' + '14.jpg')
    plt.savefig(path_blob_hist_save +os.path.sep+ 'stretch_' + '14.jpg')
    plt.close()

    result_area = 0
    result_length = 0
    result_circle = 0
    result_rect = 0
    result_stretch = 0

    nums1 = 0
    for i in range(10, 19):
        path = str(i)
        # if i == 14 or not os.path.exists('static\\images_GLCM\\images_camouflage\\mix\\20m\\' + path + '.jpg'):
        nums1 = nums1 + 1

        # img_input = cv2.imread('static\\images_GLCM\\images_camouflage\\mix\\20m\\' + path + '.jpg')
        img_input = cv2.imread(os.path.join(path1, path + '.jpg'))
        k2 = np.ones((3, 3), np.uint8)  # 开运算算子
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
        img_input = cv2.medianBlur(img_input, 5)

        img = cv2.cvtColor(img_input, cv2.COLOR_BGR2LAB)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j][0] = 0
        img1 = img

        # plt.figure("Image_0ab")
        # plt.imshow(img)

        k = 5
        dst1 = blob_kmeans(img1, k)
        dst2 = np.zeros((img1.shape), np.uint8)
        color_type = [[0, 0, 0],
                      [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255],
                      [255, 0, 255], [0, 255, 255],
                      [0, 0, 128], [0, 128, 0], [128, 0, 0], [128, 128, 0], [128, 128, 128],
                      [128, 0, 128], [0, 128, 128],
                      [0, 0, 64], [0, 64, 0], [64, 0, 0], [64, 64, 0], [64, 64, 64], [64, 0, 64], [0, 64, 64],
                      [0, 0, 192], [0, 192, 0], [192, 0, 0], [192, 192, 0], [192, 192, 192], [192, 0, 192],
                      [0, 192, 192]
                      ]

        dst3 = np.zeros((img1.shape[0], img1.shape[1], k), np.uint8)

        for i in range(k):
            point = np.argwhere(dst1 == i)
            for a, b in point:
                dst2[a, b] = color_type[i]
                dst3[a, b, i] = 255

        # cv2.imshow("result", dst2)
        # cv2.imshow("dst3", dst3[:, :, 1])
        plt.subplot(3, 3, nums1)
        # plt.figure('background')
        plt.imshow(dst2)
        # cv2.imwrite('static\\images_save\\blob_hist\\' + 'blob' + path + '.jpg', dst2)
        cv2.imwrite(path_blob_hist_save +os.path.sep+ 'blob' + path + '.jpg', dst2)

        area_ = []
        length_ = []
        # compact_ = []
        circle_ = []
        rect_ = []
        stretch_ = []  # 周长除以面积

        for n in range(k):
            dst4 = cv2.morphologyEx(dst3[:, :, n], cv2.MORPH_OPEN, k2)  # 开运算 去除细小白色区域
            # cv2.imshow("dst4", dst4)

            dst5 = cv2.morphologyEx(dst4, cv2.MORPH_CLOSE, k2)  # 开运算 去除细小白色区域
            # cv2.imshow("dst5", dst5)

            dst6 = cv2.merge((dst5, dst5, dst5))
            image, contours, hierarchy = cv2.findContours(dst5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # print('length of contours = ', len(contours))
            num_contours = len(contours)

            for i in range(num_contours):
                area_i = cv2.contourArea(contours[i])
                area_.append(area_i)

                length_i = cv2.arcLength(contours[i], True)
                length_.append(length_i)

                rect = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # print("box = ", box)
                # area_rect_i = abs(box[0][0] - box[2][0]) * abs(box[0][1] - box[2][1])
                # rect_i = area_i / area_rect_i
                # rect_.append(rect_i)
                if box[0][0] == box[2][0] or box[0][1] == box[2][1]:
                    area_rect_i = 1
                else:
                    area_rect_i = abs(box[0][0] - box[2][0]) * abs(box[0][1] - box[2][1])
                    rect_i = area_i / area_rect_i
                rect_.append(rect_i)

                circle_i = 4 * np.pi * area_i / length_i / length_i
                circle_.append(circle_i)

                stretch_i = length_i / area_i
                stretch_.append(stretch_i)

        area_ = np.array(area_, np.float32)
        length_ = np.array(length_, np.float32)
        rect_ = np.array(rect_, np.float32)
        circle_ = np.array(circle_, np.float32)
        stretch_ = np.array(stretch_, np.float32)

        array_area2 = area_
        array_length2 = length_
        array_circle2 = circle_
        array_rect2 = rect_
        array_stretch2 = stretch_

        # array_area2 = array_area2/max(array_area2)
        # array_length2 = array_length2/max(array_length2)
        # array_circle2 = array_circle2
        # array_rect2 = array_rect2
        # array_stretch2 = array_stretch2
        array_area2 = np.trunc(100 * (array_area2 / max(array_area2))) / 100
        array_length2 = np.trunc(100 * (array_length2 / max(array_length2))) / 100
        array_circle2 = np.trunc(100 * (array_circle2 / max(array_circle2))) / 100
        # print(array_circle2)
        array_rect2 = np.trunc(100 * (array_rect2 / max(array_rect2))) / 100
        array_stretch2 = np.trunc(100 * (array_stretch2 / max(array_stretch2))) / 100

        plt.figure('area_' + path)
        n, bins, patches = plt.hist(array_area2, 100, (0, 1))
        # plt.savefig('static\\images_save\\blob_hist\\' + 'area_' + path + '.jpg')
        plt.savefig(path_blob_hist_save +os.path.sep+ 'area_' + path + '.jpg')
        plt.close()

        plt.figure('length_' + path)
        n, bins, patches = plt.hist(array_length2, 100, (0, 1))
        # plt.savefig('static\\images_save\\blob_hist\\' + 'length_' + path + '.jpg')
        plt.savefig(path_blob_hist_save +os.path.sep+ 'length_' + path + '.jpg')
        plt.close()

        plt.figure('rect_' + path)
        n, bins, patches = plt.hist(array_rect2, 100, (0, 1))
        # plt.savefig('static\\images_save\\blob_hist\\' + 'rect_' + path + '.jpg')
        plt.savefig(path_blob_hist_save +os.path.sep+ 'rect_' + path + '.jpg')
        plt.close()

        plt.figure('circle_' + path)
        n, bins, patches = plt.hist(array_circle2, 100, (0, 1))
        # plt.savefig('static\\images_save\\blob_hist\\' + 'circle_' + path + '.jpg')
        plt.savefig(path_blob_hist_save +os.path.sep+ 'circle_' + path + '.jpg')
        plt.close()

        plt.figure('stretch_' + path)
        n, bins, patches = plt.hist(array_stretch2, 100, (0, 1))
        # plt.savefig('static\\images_save\\blob_hist\\' + 'stretch_' + path + '.jpg')
        plt.savefig(path_blob_hist_save +os.path.sep+ 'stretch_' + path + '.jpg')
        plt.close()

        hist_area1 = cv2.calcHist([array_area1], [0], None, [100], [0, 1])
        hist_area2 = cv2.calcHist([array_area2], [0], None, [100], [0, 1])

        # print(array_area1)
        # print(hist_area1)
        hist_length1 = cv2.calcHist([array_length1], [0], None, [100], [0, 1])
        hist_length2 = cv2.calcHist([array_length2], [0], None, [100], [0, 1])

        # print(array_length1)
        # print(hist_length1)
        hist_circle1 = cv2.calcHist([array_circle1], [0], None, [100], [0, 1])
        hist_circle2 = cv2.calcHist([array_circle2], [0], None, [100], [0, 1])

        hist_rect1 = cv2.calcHist([array_rect1], [0], None, [100], [0, 1])
        hist_rect2 = cv2.calcHist([array_rect2], [0], None, [100], [0, 1])
        # print('array_rect1 = ', array_rect1)
        # print('array_rect2 = ', array_rect2)

        hist_stretch1 = cv2.calcHist([array_stretch1], [0], None, [100], [0, 1])
        hist_stretch2 = cv2.calcHist([array_stretch2], [0], None, [100], [0, 1])

        result_area = result_area + hist_square(hist_area1, hist_area2)[6]
        result_length = result_length + hist_square(hist_length1, hist_length2)[6]
        result_circle = result_circle + hist_square(hist_circle1, hist_circle2)[6]
        result_rect = result_rect + hist_square(hist_rect1, hist_rect2)[6]
        # print(str(nums1), result_rect)
        result_stretch = result_stretch + hist_square(hist_stretch1, hist_stretch2)[6]

    # nums1 = 7
    result_area = result_area / nums1
    result_length = result_length / nums1
    result_circle = result_circle / nums1
    result_rect = result_rect / nums1
    result_stretch = result_stretch / nums1

    # print('result_area = ', result_area[0], 'result_length = ', result_length[0], 'result_circle = ', result_circle[0],
    #       'result_rect = ', result_rect[0], 'result_stretch = ', result_stretch[0])
    # plt.close()
    # plt.show()

    arr = [result_area[0], result_length[0], result_circle[0], result_rect[0], result_stretch[0]]
    # print(arr)
    sheet1.write_merge(1, 1, 1, 1, str(result_area[0]))
    sheet1.write_merge(1, 1, 2, 2, str(result_length[0]))
    sheet1.write_merge(1, 1, 3, 3, str(result_circle[0]))
    sheet1.write_merge(1, 1, 4, 4, str(result_rect[0]))
    sheet1.write_merge(1, 1, 5, 5, str(result_stretch[0]))
    # cv2.waitKey(0)
    excel_save_path = os.path.join(excels_save_blob_Kmeans, 'excel_Blob_Kmean.xls')

    f.save(excel_save_path)
    return excel_save_path



