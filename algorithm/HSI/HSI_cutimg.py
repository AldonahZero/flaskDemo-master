import cv2
import numpy as np
import matplotlib.pyplot as plt
import flaskProjectDemo.HSI as hsi
import scipy.io as io
import os


#  生成九宫格的程序  暂时用不到  暂时放在这里吧
def Hsi_cutimg_f(image_path):
    # 读取原图像
    out_path = "/image_result/" + "test"     # 这里要加一个东西用以标记是这个图片产生的结果。。。
    img_raw = hsi.load_data(image_path)
    os.mkdir(out_path)
    print(np.min(img_raw))
    print(np.max(img_raw))
    # img_raw = (img_raw - np.min(img_raw))/(np.max(img_raw) - np.min(img_raw))
    img_raw = img_raw * 255
    img_raw = img_raw.astype(np.uint8)
    # img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)

    red_band = 76
    blue_band = 15
    green_band = 46
    [m, n, p] = img_raw.shape
    print(m, n, p)
    img_r = img_raw[:, :, red_band]
    img_g = img_raw[:, :, green_band]
    img_b = img_raw[:, :, blue_band]
    img = cv2.merge([img_b, img_g, img_r])
    cv2.imwrite(out_path + 'pseudoColor.jpg', img)    # 这个函数要求路径不能有中文 否则不报错但是无结果
    # cv2.imwrite("E:/dataTest/test.png", img)
    plt.imshow(img)
    output = plt.ginput(0)
    plt.show()
    print('output = ', output)
    length_output = len(output)
    cnt = np.array(np.zeros((length_output, 1, 2)), np.float32)
    for i in range(length_output):
        # cv2.circle(img, (output[i][0].astype(int), output[i][1].astype(int)), 2, (0, 255, 255), -1)
        cnt[i, 0, 0] = output[i][0].astype(int)
        cnt[i, 0, 1] = output[i][1].astype(int)

    rect = cv2.minAreaRect(cnt)
    print('rect', rect)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    print('box = ', box)
    # cv2.imshow('img1', img)
    x, y, w, h = cv2.boundingRect(cnt)  # （x,y）是旋转的边界矩形左上角的点，w ,h分别是宽和高

    # cv2.imshow('img result', img)
    o1, o2 = rect[0]
    o1 = int(o1)
    if w < h:
        angle = int(rect[2])
        M = cv2.getRotationMatrix2D((o1, o2), angle, 1)
        rows, cols = img.shape[:2]
        print('left')
        # print(rows,cols)
        #  原图像坐标转换成 现图像坐标
        [x1, y1] = np.dot(M, np.array([[box[0][0]], [box[0][1]], [1]]))
        [x2, y2] = np.dot(M, np.array([[box[2][0]], [box[2][1]], [1]]))
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(249, 249, 3))
    else:
        angle = 90 + int(rect[2])
        M = cv2.getRotationMatrix2D((o1, o2), angle, 1)
        rows, cols = img.shape[:2]
        # print(rows,cols)
        #  原图像坐标转换成 现图像坐标
        [x1, y1] = np.dot(M, np.array([[box[0][0]], [box[0][1]], [1]]))
        [x2, y2] = np.dot(M, np.array([[box[2][0]], [box[2][1]], [1]]))
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(249, 249, 3))

    # cv2.circle(dst, (x1[0].astype(int), y1[0].astype(int)), 5, (255, 0, 255), -1)
    # cv2.circle(dst, (x2[0].astype(int), y2[0].astype(int)), 5, (255, 0, 255), -1)

    # cv2.imshow('dst', dst)

    # 左边界  有边界  上边界  下边界
    a = min(x1[0], x2[0])
    print('a', a)
    b = max(x2[0], x1[0])
    c = min(y1[0], y2[0])
    d = max(y1[0], y2[0])
    x0 = a - b + a
    x3 = b + b - a
    y0 = c - d + c
    y3 = d + d - c
    arr1 = np.array([x0, a, b, x3], np.int16)
    arr2 = np.array([y0, c, d, y3], np.int16)
    print('arr =', arr2, arr1)
    dst_h = dst.shape[1]
    dst_w = dst.shape[0]
    b = int(b)
    a = int(a)
    d = int(d)
    c = int(c)
    p = int(p)
    sample0 = np.zeros((arr2[1] - arr2[0], arr1[1] - arr1[0], p), np.uint8)
    sample1 = np.zeros((arr2[1] - arr2[0], arr1[2] - arr1[1], p), np.uint8)
    sample2 = np.zeros((arr2[1] - arr2[0], arr1[3] - arr1[2], p), np.uint8)
    sample3 = np.zeros((arr2[2] - arr2[1], arr1[1] - arr1[0], p), np.uint8)
    sample4 = np.zeros((arr2[2] - arr2[1], arr1[2] - arr1[1], p), np.uint8)
    sample5 = np.zeros((arr2[2] - arr2[1], arr1[3] - arr1[2], p), np.uint8)
    sample6 = np.zeros((arr2[3] - arr2[2], arr1[1] - arr1[0], p), np.uint8)
    sample7 = np.zeros((arr2[3] - arr2[2], arr1[2] - arr1[1], p), np.uint8)
    sample8 = np.zeros((arr2[3] - arr2[2], arr1[3] - arr1[2], p), np.uint8)
    print('d - c:', d - c)
    print('b - a:', b - a)
    print(sample0.shape)
    for p1 in range(p):
        img2 = cv2.merge([img_raw[:, :, p1], img_raw[:, :, p1], img_raw[:, :, p1]])
        dst = cv2.warpAffine(img2, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(249, 249, 3))
        for i in range(3):
            for j in range(3):
                if arr2[i] < 0 or arr2[i + 1] > dst_w or arr1[j] < 0 or arr1[j + 1] > dst_h:
                    continue
                else:
                    if dst[arr2[i], arr1[j], 0] == 249 and dst[arr2[i], arr1[j], 1] == 249 and dst[arr2[i], arr1[j], 2] == 3:
                        continue
                    if dst[arr2[i + 1], arr1[j], 0] == 249 and dst[arr2[i + 1], arr1[j], 1] == 249 and dst[arr2[i + 1], arr1[j], 2] == 3:
                        continue
                    if dst[arr2[i], arr1[j + 1], 0] == 249 and dst[arr2[i], arr1[j + 1], 1] == 249 and dst[arr2[i], arr1[j +1], 2] == 3:
                        continue
                    if dst[arr2[i + 1], arr1[j + 1], 0] == 249 and dst[arr2[i + 1], arr1[j + 1], 1] == 249 and dst[arr2[i + 1], arr1[j + 1], 2] == 3:
                        continue
                if i * 3 + j == 0:
                    temp = sample0
                elif i * 3 + j == 1:
                    temp = sample1
                elif i * 3 + j == 2:
                    temp = sample2
                elif i * 3 + j == 3:
                    temp = sample3
                elif i * 3 + j == 4:
                    temp = sample4
                elif i * 3 + j == 5:
                    temp = sample5
                elif i * 3 + j == 6:
                    temp = sample6
                elif i * 3 + j == 7:
                    temp = sample7
                else:
                    temp = sample8
                temp[:, :, p1] = dst[arr2[i].astype(int):arr2[i + 1].astype(int), arr1[j].astype(int):arr1[j + 1].astype(int), 0]

    io.savemat(out_path + '/arr0.mat', {'arr0': sample0})
    io.savemat(out_path + '/arr1.mat', {'arr1': sample1})
    io.savemat(out_path + '/arr2.mat', {'arr2': sample2})
    io.savemat(out_path + '/arr3.mat', {'arr3': sample3})
    io.savemat(out_path + '/arr4.mat', {'arr4': sample4})
    io.savemat(out_path + '/arr5.mat', {'arr5': sample5})
    io.savemat(out_path + '/arr6.mat', {'arr6': sample6})
    io.savemat(out_path + '/arr7.mat', {'arr7': sample7})
    io.savemat(out_path + '/arr8.mat', {'arr8': sample8})
