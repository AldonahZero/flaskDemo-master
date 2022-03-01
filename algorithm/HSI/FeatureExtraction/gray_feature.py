import numpy as np
import cv2
import algorithm.HSI as hsi
import algorithm.HSI.HSI_grabcut as grabcut_my


# 灰度特征


# 返回的输入矩阵的均值向量   结果是一个 1*256大小的向量
def gray_mean_f(image):
    [_, _, p] = image.shape
    result = np.zeros(p, float)
    for i in range(p):
        #  这里乘个255 让数字扩大一点
        result[i] = image[:, :, i].mean()
    return result


# 返回的输入矩阵的方差向量   结果是一个 1*256大小的向量
def gray_var_f(image):
    [_, _, p] = image.shape
    result = np.zeros(p, float)
    for i in range(p):
        result[i] = image[:, :, i].var()
    return result


# 这个要多穿进来一个 参数  用来表示 显示的是第几个波段的灰度直方图  返回的是一个  256大小的向量
def gray_histogram_f(image, band_num):
    image_gray = image[:, :, band_num] * 255.0
    image_gray = image_gray.astype(np.uint8)
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    hist = np.reshape(hist, 256)
    return hist


# 根据目标二值图生成目标矩阵   输入为高光谱原始数据  和 目标二值图
def arr_tar_f(image, image_tar):
    [m, n, p] = image.shape
    num = 0
    for i in range(m):
        for j in range(n):
            if image_tar[i, j, 0] > 127:
                num += 1
    arr = np.zeros((1, num, p), float)
    for i in range(m):
        for j in range(n):
            if image_tar[i, j, 0] > 127:
                num -= 1
                arr[0, num, :] = image[i, j, :]
    return arr


# 根据目标二值图以及灰布二值图生成背景矩阵   输入为 高光谱原始数据、目标二值图和背景二值图
def arr_back_f(image, image_tar, image_back):
    [m, n, p] = image.shape
    num = 0
    for i in range(m):
        for j in range(n):
            if image_tar[i, j, 0] < 127 and image_back[i, j, 0] < 127:
                num += 1
    arr = np.zeros((1, num, p), float)
    for i in range(m):
        for j in range(n):
            if image_tar[i, j, 0] < 127 and image_back[i, j, 0] < 127:
                num -= 1
                arr[0, num, :] = image[i, j, :]
    return arr


#  这里要传进来两个数据  一个是高光谱原始数据  一个是高光谱目标二值图
# 返回的是一个  256大小的向量  表示目标的灰度均值
def gray_mean_tar_f(image, image_tar):
    arr = arr_tar_f(image, image_tar)
    result = gray_mean_f(arr)
    return result


#  这里要传进来三个数据  一个是高光谱原始数据  一个是高光谱目标二值图  一个是高光谱图像中灰布的二值图  没有灰布则用一张全黑图像代替
#  返回的是一个  256大小的向量  表示背景的灰度均值
def gray_mean_back_f(image, image_tar, image_back):
    arr = arr_back_f(image, image_tar, image_back)
    result = gray_mean_f(arr)
    return result


#  输入：高光谱原始数据的路径
# 返回的是一个  256大小的向量  表示目标和背景的差值
#  会出现两个窗口  第一个窗口需要点点圈定目标的范围 回车 关掉窗口
#  第二个窗口圈定灰布的范围 回车，关掉窗口
def gray_mean_dif_f(image_path):
    image = hsi.load_data(image_path)
    data_path = grabcut_my.Hsi_grabcut_f(image_path)
    image_tar_path = data_path + 'target.jpg'
    image_back_path = data_path + 'back.jpg'
    image_tar = cv2.imread(image_tar_path)
    image_back = cv2.imread(image_back_path)
    [_, _, p] = image.shape
    spec_tar_mean = gray_mean_tar_f(image, image_tar)
    spec_back_mean = gray_mean_back_f(image, image_tar, image_back)
    result = np.zeros((3, p), float)
    result[0, :] = spec_tar_mean
    result[1, :] = spec_back_mean
    result[2, :] = spec_back_mean - spec_tar_mean
    return result


# 传进来两个数据  一个是高光谱原始数据  一个是目标的二值图（mask）
def gray_var_tar_f(image, image_tar):
    arr = arr_tar_f(image, image_tar)
    result = gray_var_f(arr)
    return result


#  这里要传进来三个数据  一个是高光谱原始数据  一个是高光谱目标二值图  一个是高光谱图像中灰布的二值图  没有灰布则用一张全黑图像代替
#  返回的是一个  256大小的向量  表示背景的灰度方差
def gray_var_back_f(image, image_tar, image_back):
    arr = arr_back_f(image, image_tar, image_back)
    result = gray_mean_f(arr)
    return result


#  输入：高光谱原始数据的路径
# 返回的是一个  256大小的向量  表示目标和背景的差值
#  会出现两个窗口  第一个窗口需要点点圈定目标的范围 回车 关掉窗口
#  第二个窗口圈定灰布的范围 回车，关掉窗口
def gray_var_dif_f(image_path):
    image = hsi.load_data(image_path)
    data_path = grabcut_my.Hsi_grabcut_f(image_path)
    image_tar_path = data_path + 'target.jpg'
    image_back_path = data_path + 'back.jpg'
    image_tar = cv2.imread(image_tar_path)
    image_back = cv2.imread(image_back_path)
    [_, _, p] = image.shape
    spec_tar_var = gray_var_tar_f(image, image_tar)
    spec_back_var = gray_var_back_f(image, image_tar, image_back)
    result = np.zeros((3, p), float)
    result[0, :] = spec_tar_var
    result[1, :] = spec_back_var
    result[2, :] = spec_back_var - spec_tar_var
    return result


#  输入：高光谱原始数据的路径
# 返回的是一个  256大小的向量  表示目标和背景的差值
#  会出现两个窗口  第一个窗口需要点点圈定目标的范围 回车 关掉窗口
#  第二个窗口圈定灰布的范围 回车，关掉窗口
#  注意计算时间过略长  大概10分钟以上
#  而且高光谱图像不太适合做灰度直方图  因为对于一个波段来说 大部分的像素的灰度值集中在同一个范围内 这会导致协方差系数过大
#  要想调整效果   需要再读入原始数据后  对每个波段单独图像归一化
def gray_histogram_dif_f(image_path):
    image = hsi.load_data(image_path)
    data_path = grabcut_my.Hsi_grabcut_f(image_path)
    image_tar_path = data_path + 'target.jpg'
    image_back_path = data_path + 'back.jpg'
    image_tar = cv2.imread(image_tar_path)
    image_back = cv2.imread(image_back_path)
    [_, _, p] = image.shape
    result = np.zeros(p, float)
    arr1 = arr_back_f(image, image_tar, image_back)
    arr2 = arr_tar_f(image, image_tar)
    for i in range(p):
        his_back = gray_histogram_f(arr1, i)
        his_tar = gray_histogram_f(arr2, i)
        temp = np.corrcoef(his_tar, his_back)
        result[i] = temp[0, 1]
    return result


