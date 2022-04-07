import cv2
import numpy as np
import matplotlib
from config.setting import MATPLOTLIB_INSHOW
if not MATPLOTLIB_INSHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from common.getUploadLocation import get_upload_location, get_server_location

import shutil
'''
输入：
number: images_GLCM_original 中对应的原始图像序号 测试采用 number=1
path1: 图像输入的路径
path2: 掩膜图像存储路径
path3: 分割之后的目标背景图像存储路径
output: 
[(319.49999999999994, 205.7683982683982), (329.19696969696963, 207.84632034632028), (345.1277056277056, 192.60822510822504), (370.06277056277054, 192.60822510822504), (394.99783549783547, 193.9935064935064), (426.8593073593073, 202.9978354978354), (426.8593073593073, 226.54761904761898), (422.7034632034631, 253.560606060606), (415.0844155844155, 279.1883116883116), (395.69047619047615, 281.9588744588744), (373.525974025974, 278.49567099567093), (356.20995670995666, 270.1839826839826), (346.51298701298697, 270.1839826839826), (326.4264069264069, 258.4090909090908), (324.3484848484848, 243.17099567099558), (320.88528138528136, 239.01515151515144), (295.95021645021643, 234.8593073593073)]
<class 'list'>
<class 'tuple'>
<class 'numpy.float64'>
 
输出:
一张掩膜图像存储于path2
多张图像存储于path3(1-9张)
'''
# 上传图片路径
CUTIMG_PATH = get_upload_location("/cutimg/static")
# /Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg/static
def mycutimg(img_input,path2, path3, output):
    '''
    shutil.rmtree('要清空的文件夹名')
    os.mkdir('要清空的文件夹名')
    '''
    # print(CUTIMG_PATH)
    # print(os.path.join(CUTIMG_PATH,'images_GLCM_bitwise/images_camouflage/mix/20m'))
    if (len(os.listdir(os.path.join(CUTIMG_PATH,'images_GLCM_bitwise/images_camouflage/mix/20m'))) != 0):
        # 掩膜存储路径
        shutil.rmtree(os.path.join(CUTIMG_PATH,'images_GLCM_bitwise/images_camouflage/mix/20m'))
        os.mkdir(os.path.join(CUTIMG_PATH,'images_GLCM_bitwise/images_camouflage/mix/20m'))
        # 分割之后图像存储路径
        shutil.rmtree(os.path.join(CUTIMG_PATH,'images_GLCM/images_camouflage/mix/20m'))
        os.mkdir(os.path.join(CUTIMG_PATH,'images_GLCM/images_camouflage/mix/20m'))
        # 灰度直方图存储路径
        shutil.rmtree(os.path.join(CUTIMG_PATH,'images_save/gray_histogram'))
        os.mkdir(os.path.join(CUTIMG_PATH,'images_save/gray_histogram'))
        # 边缘图像存储路径
        shutil.rmtree(os.path.join(CUTIMG_PATH,'images_GLCM_edge'))
        os.mkdir(os.path.join(CUTIMG_PATH,'images_GLCM_edge'))
        # 边缘方向直方图存储路径
        shutil.rmtree(os.path.join(CUTIMG_PATH,'images_GLCM_edge_hist'))
        os.mkdir(os.path.join(CUTIMG_PATH,'images_GLCM_edge_hist'))
        # 角点存储文件夹
        shutil.rmtree(os.path.join(CUTIMG_PATH,'images_save/coner'))
        os.mkdir(os.path.join(CUTIMG_PATH,'images_save/coner'))
        # 斑块图像存储路径
        shutil.rmtree(os.path.join(CUTIMG_PATH,'images_save/blob_hist'))
        os.mkdir(os.path.join(CUTIMG_PATH,'images_save/blob_hist'))


    img = cv2.imread(img_input)
    # path1 =
    # path2 = '/images_GLCM_bitwise/images_camouflage/mix/20m/'
    # path3 = '/images_GLCM/images_camouflage/mix/20m/'
    # 快速定位专用 实际不执行
    if False:
        os.mkdir('/images_GLCM_bitwise/images_camouflage/mix/20m/')
        os.mkdir('/images_GLCM/images_camouflage/mix/20m/')

    # img = cv2.imread(path1 + str(number) + '.JPG')
    # main
    # plt.imshow(img[:,:,::-1])
    # output = plt.ginput(0, 100)
    # print(output)
    # print(type(output))
    # print(type(output[0]))
    # print(type(output[0][0]))
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
    # cv2.imwrite('\\images_GLCM_bitwise\\images_camouflage\\mix\\20m\\' + str(number)+'.JPG', mask2)
    number = 1
    cv2.imwrite(os.path.join(path2,str(number)+'.JPG') , mask2)
    # cv2.imwrite('/' + path1 + '_bitwise' + '/images_camouflage/mix/20m/' + str(number)+'.JPG', mask2)

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
            # cv2.imwrite('/' + path1 + '_process' + '/images_camouflage/mix/20m/' + str(number) + str(i * 3 + j) + '.JPG', temp)
            # cv2.imwrite('\\images_GLCM\\images_camouflage\\mix\\20m\\' + str(number) + str(i * 3 + j) + '.JPG', temp)
            # path3 = ''
            # path_mid = path3 + str(number) + str(i * 3 + j) + '.JPG'
            cv2.imwrite(os.path.join(path3,str(number) + str(i * 3 + j) + '.JPG') , temp)
            # cv2.imwrite(path_mid, temp)

            # cv2.imshow('temp', temp)
            # cv2.waitKey(0)
    # print(path2,path3)
    return path2, path3

# mycutimg(number=1)
if __name__ == '__main__':
    img_input = 'static/images_GLCM_original/images_camouflage/mix/20m/1.JPG'
    img = cv2.imread(img_input)
    plt.imshow(img[:, :, ::-1])
    output = plt.ginput(0, 100)
    print(output)