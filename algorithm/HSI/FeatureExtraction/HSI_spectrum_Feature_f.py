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
    image_tar_path = '../image_result/target.jpg'
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

