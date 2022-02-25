import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import shutil
'''
输入：
number: images_GLCM_original 中对应的原始图像序号 测试采用 number=1
path1: 图像输入的路径
path2: 掩膜图像存储路径
path3: 分割之后的目标背景图像存储路径

输出:
一张掩膜图像存储于path2
多张图像存储于path3(1-9张)
'''

def mycutimg(img_input):
    '''
    shutil.rmtree('要清空的文件夹名')
    os.mkdir('要清空的文件夹名')
    '''

    if (len(os.listdir('static/images_GLCM_bitwise/images_camouflage/mix/20m')) != 0):
        # 掩膜存储路径
        shutil.rmtree('static/images_GLCM_bitwise/images_camouflage/mix/20m')
        os.mkdir('static/images_GLCM_bitwise/images_camouflage/mix/20m')
        # 分割之后图像存储路径
        shutil.rmtree('static/images_GLCM/images_camouflage/mix/20m')
        os.mkdir('static/images_GLCM/images_camouflage/mix/20m')
        # 灰度直方图存储路径
        shutil.rmtree('static/images_save/gray_histogram')
        os.mkdir('static/images_save/gray_histogram')
        # 边缘图像存储路径
        shutil.rmtree('static/images_GLCM_edge')
        os.mkdir('static/images_GLCM_edge')
        # 边缘方向直方图存储路径
        shutil.rmtree('static/images_GLCM_edge_hist')
        os.mkdir('static/images_GLCM_edge_hist')
        # 角点存储文件夹
        shutil.rmtree('static/images_save/coner')
        os.mkdir('static/images_save/coner')
        # 斑块图像存储路径
        shutil.rmtree('static/images_save/blob_hist')
        os.mkdir('static/images_save/blob_hist')


    img = img_input
    # path1 =
    path2 = 'static/images_GLCM_bitwise/images_camouflage/mix/20m/'
    path3 = 'static/images_GLCM/images_camouflage/mix/20m/'
    # img = cv2.imread(path1 + str(number) + '.JPG')
    # main
    plt.imshow(img[:,:,::-1])
    output = plt.ginput(0, 100)
    print(output)
    # plt.show()
    # cv2.waitKey(0)
    # print('output = ', output)
    length_output = len(output)
    cnt = np.array(np.zeros((length_output, 1, 2)), np.float32)
    for i in range(length_output):
        cnt[i, 0, 0] = output[i][0].astype(int)
        cnt[i, 0, 1] = output[i][1].astype(int)

    # 为灰度直方图等 获取掩膜
    cnt2 = np.array(cnt, np.int32)
    mask = np.zeros(img.shape, np.uint8)
    pts = cnt2.reshape((-1, 1, 2))
    # print('pts', pts)
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    # # -------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, img)
    # cv2.imwrite('D:/202177test/save/' + 'bitwise' + str(hight_name) + '/' + angle_number + '.jpg', mask2)
    # cv2.imwrite('static\\images_GLCM_bitwise\\images_camouflage\\mix\\20m\\' + str(number)+'.JPG', mask2)
    number = 1
    cv2.imwrite(path2 + str(number)+'.JPG', mask2)
    # cv2.imwrite('static/' + path1 + '_bitwise' + '/images_camouflage/mix/20m/' + str(number)+'.JPG', mask2)

    # 获取最小外接距 九宫格图像
    rect = cv2.minAreaRect(cnt)
    # print('rect', rect)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # print('box = ', box)
    # cv2.imshow('img1', img)
    x, y, w, h = cv2.boundingRect(cnt)  # （x,y）是旋转的边界矩形左上角的点，w ,h分别是宽和高

    # cv2.imshow('img result', img)
    o1, o2 = rect[0]
    o1 = int(o1)
    if w < h:
        angle = int(rect[2])
        M = cv2.getRotationMatrix2D((o1, o2), angle, 1)
        rows, cols = img.shape[:2]
        # print('left')
        #  原图像坐标转换成 现图像坐标
        [x1, y1] = np.dot(M, np.array([[box[0][0]], [box[0][1]], [1]]))
        [x2, y2] = np.dot(M, np.array([[box[2][0]], [box[2][1]], [1]]))
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(249, 249, 3))
    if w >= h:
        angle = 90 + int(rect[2])
        M = cv2.getRotationMatrix2D((o1, o2), angle, 1)
        rows, cols = img.shape[:2]
        #  原图像坐标转换成 现图像坐标
        [x1, y1] = np.dot(M, np.array([[box[0][0]], [box[0][1]], [1]]))
        [x2, y2] = np.dot(M, np.array([[box[2][0]], [box[2][1]], [1]]))
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(249, 249, 3))
        # print('right')
        # print('xy', x1, x2, y1, y2)

    # cv2.circle(dst, (x1[0].astype(int), y1[0].astype(int)), 5, (255, 0, 255), -1)
    # cv2.circle(dst, (x2[0].astype(int), y2[0].astype(int)), 5, (255, 0, 255), -1)

    # cv2.imshow('dst', dst)

    # 左边界  右边界  上边界  下边界
    a = min(x1[0], x2[0])
    b = max(x2[0], x1[0])
    c = min(y1[0], y2[0])
    d = max(y1[0], y2[0])
    x0 = a - b + a + 1
    x3 = b + b - a - 1
    y0 = c - d + c + 1
    y3 = d + d - c - 1
    arr1 = np.array([x0, a, b, x3], np.int16)
    arr2 = np.array([y0, c, d, y3], np.int16)
    dst_h = dst.shape[1]
    dst_w = dst.shape[0]

    # 创建存储文件夹
    # if os.path.exists('D:/202177test/save/' + str(hight_name)) is False:
    #     os.mkdir('D:/202177test/save/' + str(hight_name))
    for i in range(3):
        for j in range(3):
            if arr2[i] < 0 or arr2[i + 1] > dst_w or arr1[j] < 0 or arr1[j + 1] > dst_h:
                continue
            else:
                if dst[arr2[i], arr1[j], 0] == 249 and dst[arr2[i], arr1[j], 1] == 249 and dst[
                    arr2[i], arr1[j], 2] == 3:
                    continue
                if dst[arr2[i + 1], arr1[j], 0] == 249 and dst[arr2[i + 1], arr1[j], 1] == 249 and dst[
                    arr2[i + 1], arr1[j], 2] == 3:
                    continue
                if dst[arr2[i], arr1[j + 1], 0] == 249 and dst[arr2[i], arr1[j + 1], 1] == 249 and dst[
                    arr2[i], arr1[j + 1], 2] == 3:
                    continue
                if dst[arr2[i + 1], arr1[j + 1], 0] == 249 and dst[arr2[i + 1], arr1[j + 1], 1] == 249 and dst[
                    arr2[i + 1], arr1[j + 1], 2] == 3:
                    continue
            temp = dst[arr2[i].astype(int):arr2[i + 1].astype(int), arr1[j].astype(int):arr1[j + 1].astype(int), :]

            # cv2.imwrite('D:/202177test/save/' + str(hight_name) + '/' + str(angle_number) + str(i * 3 + j) + '.jpg',
            #             temp)
            # cv2.imwrite('static/' + path1 + '_process' + '/images_camouflage/mix/20m/' + str(number) + str(i * 3 + j) + '.JPG', temp)
            # cv2.imwrite('static\\images_GLCM\\images_camouflage\\mix\\20m\\' + str(number) + str(i * 3 + j) + '.JPG', temp)
            # path3 = ''
            # path_mid = path3 + str(number) + str(i * 3 + j) + '.JPG'
            cv2.imwrite(path3 + str(number) + str(i * 3 + j) + '.JPG', temp)
            # cv2.imwrite(path_mid, temp)

            # cv2.imshow('temp', temp)
            # cv2.waitKey(0)
    print(path2,path3)
    return path2, path3

# mycutimg(number=1)
