import cv2
import numpy as np
import matplotlib.pyplot as plt
import algorithm.HSI as hsi
from config.setting import RESULT_FOLDER


# 图像分割程序   输入为高光谱图像原始数据  返回值为结果保存目录   结果自动保存为该目录下的  target.jpg  back.jpg文件
# 修改输出路径  直接改 out_path 即可
def Hsi_grabcut_f(image_path):
    # 读取原图像
    out_path = RESULT_FOLDER + '/hsi' + image_path[image_path.rindex('/'):image_path.rindex('.')]

    img_raw = hsi.load_data(image_path)
    # os.mkdir(out_path)   # 创建文件夹的作用  如果已经有所有的对应文件夹后  注释掉  如果没有  加上
    # print(np.min(img_raw))
    # print(np.max(img_raw))
    # img_raw = (img_raw - np.min(img_raw))/(np.max(img_raw) - np.min(img_raw))
    img_raw = img_raw * 255
    img_raw = img_raw.astype(np.uint8)
    # img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)

    red_band = 76
    blue_band = 15
    green_band = 46
    [m, n, p] = img_raw.shape
    # print(m, n, p)
    img_r = img_raw[:, :, red_band]
    img_g = img_raw[:, :, green_band]
    img_b = img_raw[:, :, blue_band]
    img = cv2.merge([img_b, img_g, img_r])

    out_path1 = out_path + 'target.jpg'
    img2 = (img - np.min(img)) / (np.max(img) - np.min(img))
    # img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    plt.figure(figsize=(12, 9))  # 自己设定窗口图片的大小
    plt.imshow(img2[:, :, [2, 1, 0]])
    output = plt.ginput(0)
    plt.show()
    # print('output = ', output)
    length_output = len(output)
    cnt = np.array(np.zeros((length_output, 1, 2)), np.int32)

    # plt.fill()
    x = np.array(np.zeros(length_output), np.int16)
    y = np.array(np.zeros(length_output), np.int16)
    for i in range(length_output):
        # cv2.circle(img, (output[i][0].astype(int), output[i][1].astype(int)), 2, (0, 255, 255), -1)
        cnt[i, 0, 0] = output[i][0].astype(int)
        cnt[i, 0, 1] = output[i][1].astype(int)
        x[i] = cnt[i, 0, 0]
        y[i] = cnt[i, 0, 1]
    '''
    plt.fill(x, y, 'g', 1)
    '''

    mask = np.zeros(img.shape, np.uint8)
    pts = cnt.reshape((-1, 1, 2))
    # print('pts', pts)
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    # # -------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imwrite(out_path1, mask2)
    cv2.imshow('mask', mask2)
    cv2.waitKey(0)

    out_path2 = out_path + 'back.jpg'

    img2 = (img - np.min(img)) / (np.max(img) - np.min(img))
    # img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    plt.figure(figsize=(12, 9))  # 自己设定窗口图片的大小
    plt.imshow(img2[:, :, [2, 1, 0]])
    output = plt.ginput(0)
    plt.show()
    # print('output = ', output)
    length_output = len(output)
    cnt = np.array(np.zeros((length_output, 1, 2)), np.int32)

    # plt.fill()
    x = np.array(np.zeros(length_output), np.int16)
    y = np.array(np.zeros(length_output), np.int16)
    for i in range(length_output):
        # cv2.circle(img, (output[i][0].astype(int), output[i][1].astype(int)), 2, (0, 255, 255), -1)
        cnt[i, 0, 0] = output[i][0].astype(int)
        cnt[i, 0, 1] = output[i][1].astype(int)
        x[i] = cnt[i, 0, 0]
        y[i] = cnt[i, 0, 1]
    '''
    plt.fill(x, y, 'g', 1)
    '''

    mask = np.zeros(img.shape, np.uint8)
    pts = cnt.reshape((-1, 1, 2))
    # print('pts', pts)
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    # # -------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imwrite(out_path2, mask2)
    cv2.imshow('mask', mask2)
    #  cv2.imshow('ROI', ROI)
    cv2.waitKey(0)
    return out_path

