import numpy as np
import cv2
import matplotlib.pyplot as plt
import algorithm.HSI as hsi
import scipy.io as io


# 角点检测显示程序
# 输入为高光谱原始数据 输出为一幅图像 返回值为图像默认保存路径

def Harris_points_f(mat_path, out_path):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat(mat_path)
    image = data['arr4']
    red_band = 76
    blue_band = 15
    green_band = 46
    img_r = image[:, :, red_band]
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band]
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band]
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    img = image[:, :, 170]    #   波段索引是写死的  这个是根据实验得到的数据
    [m, n] = img.shape
    print([m, n])
    img = np.float32(img)
    dst = cv2.cornerHarris(img, 2, 3, 0, 0.4)
    dst = cv2.dilate(dst, None)  # 角点原来是个小叉叉（××） 膨胀角点
    pseudo_image[dst > (0.1 * dst.max())] = [0, 0, 255]
    # out_path = "../image_result/points_Harris.jpg"
    cv2.imwrite(out_path, pseudo_image)
    return out_path


def SURF_points_f(mat_path, out_path):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat(mat_path)
    image = data['arr4']
    red_band = 76
    blue_band = 15
    green_band = 46
    img_r = image[:, :, red_band]
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band]
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band]
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    # img = image[:, :, 30]  # 波段索引是写死的  这个是根据实验得到的数据
    img = cv2.cvtColor(pseudo_image, cv2.COLOR_BGR2GRAY)
    # img = np.float32(img)
    sift = cv2.xfeatures2d_SURF.create()   # 这个程序要求和齐王靖那个一样的opencv版本   我电脑上没有  所以没做结果。。。
    kp, des = sift.detectAndCompute(img, None)
    kp_image = cv2.drawKeypoints(pseudo_image, kp, None, color=(0, 0, 255))
    # out_path = "../image_result/points_SURF.jpg"
    cv2.imwrite(out_path, kp_image)
    return out_path


def SIFT_points_f(mat_path, out_path):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat(mat_path)
    image = data['arr4']
    red_band = 76
    blue_band = 15
    green_band = 46
    img_r = image[:, :, red_band]
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band]
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band]
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    #img = image[:, :, 30]  # 波段索引是写死的  这个是根据实验得到的数据
    img = cv2.cvtColor(pseudo_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()    # 这个程序要求和齐王靖那个一样的opencv版本   我电脑上没有  所以没做结果。。。
    kp, des1 = sift.detectAndCompute(img, None)
    kp_image = cv2.drawKeypoints(pseudo_image, kp, None, color=(0, 0, 255))
    # out_path = "../image_result/points_SIFT.jpg"
    cv2.imwrite(out_path, kp_image)
    return out_path


def Fast_points_f(mat_path, out_path):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat(mat_path)
    image = data['arr4']
    red_band = 76
    blue_band = 15
    green_band = 46
    print(1)
    img_r = image[:, :, red_band]
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band]
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band]
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    img = image[:, :, 100]  # 波段索引是写死的  这个是根据实验得到的数据
    # img = np.float32(img)
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    kp_image = cv2.drawKeypoints(pseudo_image, kp, None, color=(0, 0, 255))
    # out_path = "../image_result/points_Fast.jpg"
    cv2.imwrite(out_path, kp_image)
    return out_path


def ORB_points_f(mat_path, out_path):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat(mat_path)
    image = data['arr4']
    red_band = 76
    blue_band = 15
    green_band = 46
    img_r = image[:, :, red_band]
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band]
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band]
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    img = image[:, :, 100]  # 波段索引是写死的  这个是根据实验得到的数据
    # img = np.float32(img)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img, None)
    kp_image = cv2.drawKeypoints(pseudo_image, kp1, None, color=(0, 0, 255))
    # out_path = "../image_result/points_ORB.jpg"
    cv2.imwrite(out_path, kp_image)
    return out_path


def KAZE_points_f(mat_path, out_path):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat(mat_path)
    image = data['arr4']
    red_band = 76
    blue_band = 15
    green_band = 46
    img_r = image[:, :, red_band]
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band]
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band]
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    img = image[:, :, 100]  # 波段索引是写死的  这个是根据实验得到的数据
    # img = np.float32(img)

    fast = cv2.KAZE_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = fast.detect(img, None)
    kp1, des1 = brief.compute(img, kp)
    kp_image = cv2.drawKeypoints(pseudo_image, kp1, None, color=(0, 0, 255))
    # out_path = "../image_result/points_KAZE.jpg"
    cv2.imwrite(out_path, kp_image)
    return out_path


# Harris_points_f("../image_test/bwz/半伪装146.raw")