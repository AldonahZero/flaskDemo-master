import uuid
from datetime import datetime

import numpy as np
import cv2
import algorithm.HSI as hsi
import algorithm.HSI.HSI_grabcut as grabcut_my


# 灰度特征


# 返回的输入矩阵的均值向量   结果是一个 1*256大小的向量
from common.mysql_operate import db_session, HSIResultFile


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
# 返回的是一个  256大小的向量  表示目标均值
# 灰度均值
def gray_mean_dif_f(image_path, data_path, excel_save_path):
    file_name = image_path.split("/")[-1]
    key = file_name[0:file_name.rindex('.')]

    image = hsi.load_data(image_path)
    image_tar_path = data_path + 'target.jpg'
    image_back_path = data_path + 'back.jpg'
    image_tar = cv2.imread(image_tar_path)
    image_back = cv2.imread(image_back_path)
    [_, _, p] = image.shape
    spec_tar_mean = gray_mean_tar_f(image, image_tar)
    spec_back_mean = gray_mean_back_f(image, image_tar, image_back)
    spec_mean_dif = spec_back_mean - spec_tar_mean

    # 输出结果
    excel_id = str(uuid.uuid1())
    out_path = excel_save_path + excel_id + '_result_mean.xls'
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

    # 将结果记录存放到数据库
    session = db_session()
    excel_result_file = HSIResultFile(fid=excel_id, pid=key, type="gray_mean_dif_f", path=out_path, create_time=datetime.now())
    session.add(excel_result_file)
    session.commit()
    session.close()

    return spec_tar_mean


#  输入：高光谱原始数据的路径
# 返回的是一个  256大小的向量  表示目标方差

def gray_var_dif_f(image_path, data_path, excel_save_path):
    file_name = image_path.split("/")[-1]
    key = file_name[0:file_name.rindex('.')]
    image = hsi.load_data(image_path)
    image_tar_path = data_path + 'target.jpg'
    image_back_path = data_path + 'back.jpg'
    image_tar = cv2.imread(image_tar_path)
    image_back = cv2.imread(image_back_path)
    [_, _, p] = image.shape
    spec_tar_var = gray_var_tar_f(image, image_tar)
    spec_back_var = gray_var_back_f(image, image_tar, image_back)
    spec_var_dif = spec_back_var - spec_tar_var
    # 输出结果
    excel_id = str(uuid.uuid1())
    out_path = excel_save_path + excel_id + '_result_var.xls'
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

    # 将结果记录存放到数据库
    session = db_session()
    excel_result_file = HSIResultFile(fid=excel_id, pid=key, type="gray_var_dif_f", path=out_path,
                                      create_time=datetime.now())
    session.add(excel_result_file)
    session.commit()
    session.close()

    return spec_tar_var


#  输入：高光谱原始数据的路径
# 返回的是一个  256大小的向量  表示目标的灰度直方图

def gray_histogram_dif_f(image_path, band_index, data_path, excel_save_path):
    file_name = image_path.split("/")[-1]
    key = file_name[0:file_name.rindex('.')]
    image = hsi.load_data(image_path)
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

    excel_id = str(uuid.uuid1())
    out_path = excel_save_path + excel_id + 'result_hist.xls'
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

    # 将结果记录存放到数据库
    session = db_session()
    excel_result_file = HSIResultFile(fid=excel_id, pid=key, type="gray_histogram_dif_f", path=out_path,
                                      create_time=datetime.now())
    session.add(excel_result_file)
    session.commit()
    session.close()

    return his_tar

