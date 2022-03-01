import numpy as np
import cv2
import matplotlib.pyplot as plt
import algorithm.HSI as hsi
import algorithm.HSI.FeatureExtraction as Feature_my


#  光谱特征

#  高光谱光谱特征提取算法 输入raw格式高光谱原始数据路径
#  输出一个概率图

#  归一化水体指数  用来衡量该像素点对应的地物与水体的接近程序（ 镜面反射 以及 水汽的含量 都会导致其接近水体）
#  返回一个二维矩阵  大小为 -1 到 1（ 可以直接显示  不用考虑负数）
def HSI_NDWI_f(image_path):
    image = hsi.load_data(image_path)
    result = (image[:, :, 46] - image[:, :, 167]) / (image[:, :, 46] + image[:, :, 167])
    # cv2.imshow("RESULT", result)
    # cv2.waitKey(0)
    out_path = "../image_result/NDWI_result.jpg"
    result = result * 255
    cv2.imwrite(out_path, result)
    return out_path


#  归一化植被指数  用来衡量该像素点对应的地物与植被的接近程序（叶绿素含量）
#  返回一个路径  其实不用返回  固定在当前HSI包下面的image_result文件夹中
def HSI_NDVI_f(image_path):
    image = hsi.load_data(image_path)
    result = (image[:, :, 167] - image[:, :, 76]) / (image[:, :, 167] + image[:, :, 76])
    # cv2.imshow("RESULT", result)
    # cv2.waitKey(0)
    out_path = "../image_result/NDVI_result.jpg"
    result = result * 255
    cv2.imwrite(out_path, result)
    return out_path


#  根据光谱角距离得到地物接近目标光谱的 概率(两向量之间的反余弦值)。
#  输入 高光谱数据的路径： image_path     目标的二值图（通过 grabcut 程序得到的分割图像）
def HSI_SAM_f(image_path):
    image = hsi.load_data(image_path)
    image_tar_path = HSI_SAM_grabcut_f(image)
    image_tar = cv2.imread(image_tar_path)
    [m, n, p] = image.shape
    spec_mean = Feature_my.gray_mean_tar_f(image, image_tar)
    spec_mean = spec_mean.reshape(p, 1)
    result = np.zeros((m, n), float)
    for i in range(m):
        for j in range(n):
            temp = image[i, j, :].reshape(1, p)
            result[i, j] = np.arccos(np.dot(temp, spec_mean) / (np.linalg.norm(spec_mean) * np.linalg.norm(temp)))
    # cv2.imshow("RESULT", result)
    # cv2.waitKey(0)
    out_path = "../image_result/SAM_result.jpg"
    result = result * 255
    cv2.imwrite(out_path, result)
    return out_path


def HSI_SAM_grabcut_f(img_raw):
    # 读取原图像
    out_path = '../image_result/'
    # os.mkdir(out_path)   # 创建文件夹的作用  如果已经有所有的对应文件夹后  注释掉  如果没有  加上
    # img_raw = (img_raw - np.min(img_raw))/(np.max(img_raw) - np.min(img_raw))
    img_raw = img_raw * 255
    img_raw = img_raw.astype(np.uint8)
    # img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)

    red_band = 76
    blue_band = 15
    green_band = 46
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
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    # # -------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imwrite(out_path1, mask2)
    return out_path1


