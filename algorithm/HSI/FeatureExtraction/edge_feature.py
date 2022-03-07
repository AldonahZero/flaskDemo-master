import cv2
import algorithm.HSI as hsi
import numpy as np
from skimage import filters


def image_normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))
    image = (image - mean) / np.sqrt(var)
    return image


def gauss_edge_f(image_path, k_num):
    image = hsi.load_data(image_path)
    img = image[:, :, k_num]
    img = image_normalize(img)
    img = img * 255
    img = img.astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gauss_img = img - blurred
    #  输出路径
    out_path = "../image_result/edge_gauss_result.jpg"
    cv2.imwrite(out_path, gauss_img)
    return out_path


def canny_edge_f(image_path, k_num, out_path):
    image = hsi.load_data(image_path)
    img = image[:, :, k_num]
    img = image_normalize(img)
    img = img * 255
    img = img.astype(np.uint8)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    canny_image = cv2.Canny(blurred, 50, 150)

    #  输出路径
    # out_path = "../image_result/edge_canny_result.jpg"
    cv2.imwrite(out_path, canny_image)
    return out_path


# Laplace算子
# 常用的Laplace算子模板  [[0, 1, 0], [1, -4, 1], [0, 1, 0]]   [[1 ,1 , 1],[1, -8, 1],[1, 1, 1]]
def laplace_edge_f(image_path, k_num):
    image = hsi.load_data(image_path)
    img = image[:, :, k_num]
    img = image_normalize(img)
    img = img * 255
    img = img.astype(np.uint8)
    laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    laplace_image = cv2.convertScaleAbs(laplacian)

    out_path = '../image_result/edge_Laplace_result.jpg'
    cv2.imwrite(out_path, laplace_image)
    return out_path


def prewitt_edge_f(image_path, k_num):
    image = hsi.load_data(image_path)
    img = image[:, :, k_num]
    img = image_normalize(img)
    img = img * 255
    img = img.astype(np.uint8)

    # img = np.reshape(img, (m, n))

    # Prewitt 算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)

    # 转 uint8 ,图像融合
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    out_path = "../image_result/edge_Prewitt_result.jpg"
    cv2.imwrite(out_path, Prewitt)
    return out_path


def sobel_edge_f(image_path, k_num):
    image = hsi.load_data(image_path)
    [m, n, p] = image.shape
    img = image[:, :, k_num] * 255
    img = np.reshape(img, (m, n))
    r, c = img.shape
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    out_path = "../image_result/edge_soble_result.jpg"
    cv2.imwrite(out_path, result)
    return out_path


def roberts_edge_f(image_path, k_num):
    image = hsi.load_data(image_path)
    img = image[:, :, k_num]
    img = image_normalize(img)
    img = img * 255
    img = img.astype(np.uint8)
    roberts = filters.roberts(img)
    roberts = roberts * 255.0
    out_path = "../image_result/edge_roberts_result.jpg"
    cv2.imwrite(out_path, roberts)
    return out_path

