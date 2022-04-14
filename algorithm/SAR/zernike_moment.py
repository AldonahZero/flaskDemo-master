import csv
import os
import math
import numpy as np
from config.setting import RESULT_FOLDER
from .process_pre import otsu_2d


##计算径向多项式公式
def radialpoly(r, n, m):
    [n1, m1] = r.shape
    rad = np.zeros((n1, m1))
    num = 1 + (n - abs(m)) / 2
    num = int(num)
    for s in range(num):
        c = (-1) ** s * math.factorial(n - s) / (math.factorial(s) * math.factorial((n + abs(m)) / 2 - s)
                                                 * math.factorial((n - abs(m)) / 2 - s))
        rad = rad + c * r ** (n - 2 * s)
    return rad


##计算图像zernike矩函数
def Zernike_moment(img, n, m):
    '''
    :param img: 输入图像
    :param n,m: n,m为正交多项式阶数，n非负整数，n-|m|为非负偶数。
    :return:
    '''
    # img为图像，n，m为阶数
    m00 = img.sum()
    m_p, n_p = img.shape
    if m00 == 0:
        m00 = math.exp(-20)
    y = np.linspace(1, m_p, m_p)
    x = np.linspace(1, n_p, n_p)
    X, Y = np.meshgrid(x, y)
    X = X / n_p
    Y = Y / m_p
    m10 = np.vdot(X, img)
    m01 = np.vdot(Y, img)
    xc = m10 / m00
    yc = m01 / m00
    X = X - xc
    Y = Y - yc
    # R = np.sqrt(np.power(X,2)+np.power(Y,2))
    num = X ** 2 + Y ** 2
    R = np.sqrt(num)
    theta = np.arctan2(Y, X)
    theta = (R <= 1) * theta
    R = (R <= 1) * R  # 定义单位圆
    Rad = radialpoly(R, n, m)
    [k1, k2] = theta.shape
    exp_r = np.cos(m * theta)
    exp_i = np.sin(m * theta)
    exp_m = exp_r - 1j * exp_i  # 构造exp函数
    Product = img * Rad * exp_m
    Z = sum(Product.reshape(-1))
    Z = (n + 1) * Z / math.pi
    Z = Z / m00
    A = abs(Z)
    return A


def get_zernike(path, n=2, m=2):
    '''
    :param path: 输入图像存储路径
    :param n,m: n,m为正交多项式阶数，n非负整数，n-|m|为非负偶数。默认n=m=2
    :return: 计算八个zernike矩的存储路径
    '''
    _, img = otsu_2d(path)
    img = img / 255
    zernike_list = []
    zernike_name = []
    # n = 2
    # m = 2
    for i in range(8):
        c = Zernike_moment(img, n, m)
        zernike_list.append(c)
        zernike_name.append('Z' + str(n) + str(m))
        n += 2
    # zernike_path = RESULT_FOLDER + '/SAR/Zernike.csv'
    zernike_path = os.path.join(RESULT_FOLDER, 'SAR/zernike.csv')
    f = open(zernike_path, 'w', newline="")
    csv_writer = csv.writer(f)
    # 构建列表头
    csv_writer.writerow(zernike_name)
    # 写入csv文件内容
    csv_writer.writerow(zernike_list)
    f.close()
    return zernike_path, zernike_list

# if __name__ == '__main__':
#     print(get_zernike(r'D:\back_dev_flask-master\static\result\SAR\image_b.png'))
