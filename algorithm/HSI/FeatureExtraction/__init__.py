import cv2
import algorithm.HSI as hsi
import numpy as np
import matplotlib.pyplot as plt
import algorithm.HSI.HSI_grabcut as grabcut_my
from skimage import filters
import scipy.io as io


#  边缘特征


def image_normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))
    image = (image - mean) / np.sqrt(var)
    return image


def gauss_edge_f(image_path, k_num):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat('../image_result/arr4.mat')
    image = data['arr4']

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


def canny_edge_f(image_path, k_num):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat('../image_result/arr4.mat')
    image = data['arr4']
    '''red_band = 76
    blue_band = 15
    green_band = 46
    img_r = image[:, :, red_band] * 255
    img_r = img_r.astype(np.uint8)
    img_g = image[:, :, green_band] * 255
    img_g = img_g.astype(np.uint8)
    img_b = image[:, :, blue_band] * 255
    img_b = img_b.astype(np.uint8)
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    cv2.imshow("test", pseudo_image)
    cv2.waitKey(0)'''

    img = image[:, :, k_num]
    img = image_normalize(img)
    img = img * 255
    img = img.astype(np.uint8)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    canny_image = cv2.Canny(blurred, 50, 150)

    #  输出路径
    out_path = "../image_result/edge_canny_result.jpg"
    cv2.imwrite(out_path, canny_image)
    return out_path


# Laplace算子
# 常用的Laplace算子模板  [[0, 1, 0], [1, -4, 1], [0, 1, 0]]   [[1 ,1 , 1],[1, -8, 1],[1, 1, 1]]
def laplace_edge_f(image_path, k_num):
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat('../image_result/arr4.mat')
    image = data['arr4']
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
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat('../image_result/arr4.mat')
    image = data['arr4']
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
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat('../image_result/arr4.mat')
    image = data['arr4']
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
    #  image = hsi.load_data(image_path)
    #  暂时直接拿的图像分割的结果来做的不需要源地址
    data = io.loadmat('../image_result/arr4.mat')
    image = data['arr4']
    img = image[:, :, k_num]
    img = image_normalize(img)
    img = img * 255
    img = img.astype(np.uint8)
    roberts = filters.roberts(img)
    roberts = roberts * 255.0
    out_path = "../image_result/edge_roberts_result.jpg"
    cv2.imwrite(out_path, roberts)
    return out_path


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
def gray_mean_dif_f(image_path):
    image = hsi.load_data(image_path)
    data_path = '../image_result/'
    image_tar_path = data_path + 'target.jpg'
    image_back_path = data_path + 'back.jpg'
    image_tar = cv2.imread(image_tar_path)
    image_back = cv2.imread(image_back_path)
    [_, _, p] = image.shape
    spec_tar_mean = gray_mean_tar_f(image, image_tar)
    spec_back_mean = gray_mean_back_f(image, image_tar, image_back)
    spec_mean_dif = spec_back_mean - spec_tar_mean

    # 输出结果

    out_path = '../image_result/result_mean.xls'
    output = open(out_path, 'w', encoding='gbk')
    output.write("目标灰度均值")
    output.write('\t')
    for i in range(p):
        output.write(str(spec_tar_mean[i]))
        output.write('\t')
    output.write('\n')

    output.write("背景灰度均值")
    output.write('\t')
    for i in range(p):
        output.write(str(spec_back_mean[i]))
        output.write('\t')
    output.write('\n')

    output.write("目标背景灰度均值之差")
    output.write('\t')
    for i in range(p):
        output.write(str(spec_mean_dif[i]))
        output.write('\t')
    return out_path


#  输入：高光谱原始数据的路径
# 返回的是一个  256大小的向量  表示目标和背景的差值
#  会出现两个窗口  第一个窗口需要点点圈定目标的范围 回车 关掉窗口
#  第二个窗口圈定灰布的范围 回车，关掉窗口
def gray_var_dif_f(image_path):
    image = hsi.load_data(image_path)
    data_path = '../image_result/'
    image_tar_path = data_path + 'target.jpg'
    image_back_path = data_path + 'back.jpg'
    image_tar = cv2.imread(image_tar_path)
    image_back = cv2.imread(image_back_path)
    [_, _, p] = image.shape
    spec_tar_var = gray_var_tar_f(image, image_tar)
    spec_back_var = gray_var_back_f(image, image_tar, image_back)
    spec_var_dif = spec_back_var - spec_tar_var
    # 输出结果

    out_path = '../image_result/result_var.xls'
    output = open(out_path, 'w', encoding='gbk')
    output.write("目标灰度方差")
    output.write('\t')
    for i in range(p):
        output.write(str(spec_tar_var[i]))
        output.write('\t')
    output.write('\n')

    output.write("背景灰度方差")
    output.write('\t')
    for i in range(p):
        output.write(str(spec_back_var[i]))
        output.write('\t')
    output.write('\n')

    output.write("目标背景灰度方差之差")
    output.write('\t')
    for i in range(p):
        output.write(str(spec_var_dif[i]))
        output.write('\t')
    return out_path


#  输入：高光谱原始数据的路径
# 返回的是一个  256大小的向量  表示目标和背景的差值
#  会出现两个窗口  第一个窗口需要点点圈定目标的范围 回车 关掉窗口
#  第二个窗口圈定灰布的范围 回车，关掉窗口
#  注意计算时间过略长  大概10分钟以上
#  而且高光谱图像不太适合做灰度直方图  因为对于一个波段来说 大部分的像素的灰度值集中在同一个范围内 这会导致协方差系数过大
#  要想调整效果   需要再读入原始数据后  对每个波段单独图像归一化
def gray_histogram_dif_f(image_path, band_index):
    image = hsi.load_data(image_path)
    data_path = '../image_result/'
    image_tar_path = data_path + 'target.jpg'
    image_back_path = data_path + 'back.jpg'
    image_tar = cv2.imread(image_tar_path)
    image_back = cv2.imread(image_back_path)
    [_, _, p] = image.shape
    arr1 = arr_tar_f(image, image_tar)
    arr2 = arr_back_f(image, image_tar, image_back)
    his_tar = gray_histogram_f(arr1, band_index)
    his_back = gray_histogram_f(arr2, band_index)
    # 输出结果

    out_path = '../image_result/result_hist.xls'
    output = open(out_path, 'w', encoding='gbk')
    output.write("目标灰度直方图")
    output.write('\t')
    for i in range(256):
        output.write(str(his_tar[i]))
        output.write('\t')
    output.write('\n')

    output.write("背景灰度直方图")
    output.write('\t')
    for i in range(256):
        output.write(str(his_back[i]))
        output.write('\t')
    output.write('\n')

    return out_path


#  光谱特征

#  高光谱光谱特征提取算法 输入raw格式高光谱原始数据路径
#  输出一个概率图

#  归一化水体指数  用来衡量该像素点对应的地物与水体的接近程序（ 镜面反射 以及 水汽的含量 都会导致其接近水体）
#  返回一个二维矩阵  大小为 -1 到 1（ 可以直接显示  不用考虑负数）
def HSI_NDWI_f(image_path, out_path):
    image = hsi.load_data(image_path)
    result = (image[:, :, 46] - image[:, :, 167]) / (image[:, :, 46] + image[:, :, 167])
    # cv2.imshow("RESULT", result)
    # cv2.waitKey(0)
    # out_path = "../image_result/NDWI_result.jpg"
    result = result * 255
    cv2.imwrite(out_path, result)
    return out_path


#  归一化植被指数  用来衡量该像素点对应的地物与植被的接近程序（叶绿素含量）
#  返回一个路径  其实不用返回  固定在当前HSI包下面的image_result文件夹中
def HSI_NDVI_f(image_path, out_path):
    image = hsi.load_data(image_path)
    result = (image[:, :, 167] - image[:, :, 76]) / (image[:, :, 167] + image[:, :, 76])
    # cv2.imshow("RESULT", result)
    # cv2.waitKey(0)
    # out_path = "../image_result/NDVI_result.jpg"
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
    spec_mean = gray_mean_tar_f(image, image_tar)
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


# 角点检测显示程序
# 输入为高光谱原始数据 输出为一幅图像 返回值为图像默认保存路径

def Harris_points_f(image_path, out_path):
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
    # out_path = "../image_result/points_Harris.jpg"
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
