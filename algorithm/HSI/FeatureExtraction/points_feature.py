import numpy as np
import cv2
import matplotlib.pyplot as plt
import algorithm.HSI as hsi


# 角点检测显示程序
# 输入为高光谱原始数据 输出为一幅图像 返回值为图像默认保存路径

def Harris_points_f(image_path):
    image = hsi.load_data(image_path)
    red_band = 76
    blue_band = 15
    green_band = 46
    img_r = image[:, :, red_band] * 255
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band] * 255
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band] * 255
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    img = image[:, :, 170]    #   波段索引是写死的  这个是根据实验得到的数据
    [m, n] = img.shape
    print([m, n])
    img = np.float32(img)
    dst = cv2.cornerHarris(img, 2, 3, 0, 0.4)
    dst = cv2.dilate(dst, None)  # 角点原来是个小叉叉（××） 膨胀角点
    pseudo_image[dst > 0.01 * dst.max()] = [0, 0, 255]
    out_path = "../image_result/points_Harris.jpg"
    cv2.imwrite(out_path, pseudo_image)
    return out_path


def SURF_points_f(image_path):
    image = hsi.load_data(image_path)
    red_band = 76
    blue_band = 15
    green_band = 46
    img_r = image[:, :, red_band] * 255
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band] * 255
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band] * 255
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    img = image[:, :, 30]  # 波段索引是写死的  这个是根据实验得到的数据
    img = np.float32(img)
    sift = cv2.xfeatures2d_SIFT.create()   # 这个程序要求和齐王靖那个一样的opencv版本   我电脑上没有  所以没做结果。。。
    kp, des = sift.detectAndCompute(img, None)
    kp_image = cv2.drawKeypoints(pseudo_image, kp, None, color=(0, 0, 255))
    out_path = "../image_result/points_Harris.jpg"
    cv2.imwrite(out_path, kp_image)
    return out_path

