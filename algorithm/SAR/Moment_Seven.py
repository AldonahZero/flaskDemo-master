import csv
import os
import math
import numpy as np
from .process_pre import otsu_2d

RESULT_FOLDER = os.path.join('algorithm', 'SAR', 'result')


def MomentSeven(path):
    '''
    :param img: 输入图像
    :return:
    '''
    _,otsuimage = otsu_2d(path)
    m, n = otsuimage.shape
    m00 = otsuimage.sum()
    if m00 == 0:
        m00 = math.exp(-20)
    x1 = np.linspace(1, m, m)
    y1 = np.linspace(1, n, n)
    X, Y = np.meshgrid(x1, y1)

    m10 = np.vdot(X, otsuimage)
    m01 = np.vdot(Y, otsuimage)
    x_mean = m10 / m00
    y_mean = m01 / m00
    k_y = Y - y_mean
    k_x = X - x_mean
    k_x2 = k_x * k_x
    k_y2 = k_y * k_y
    k_x3 = k_x * k_x * k_x
    k_y3 = k_y * k_y * k_y

    cm00 = float(m00)
    cm02 = (np.vdot(k_y2, otsuimage)) / m00 ** 2
    cm03 = (np.vdot(k_y3, otsuimage)) / m00 ** 2.5
    cm11 = (np.vdot(k_x * k_y, otsuimage)) / m00 ** 2
    cm12 = (np.vdot(k_x * k_y2, otsuimage)) / m00 ** 2.5
    cm20 = (np.vdot(k_x2, otsuimage)) / m00 ** 2
    cm21 = (np.vdot(k_x2 * k_y, otsuimage)) / m00 ** 2.5
    cm30 = (np.vdot(k_x3, otsuimage)) / m00 ** 2.5

    Mon = np.zeros(10)
    # 1~7阶矩
    Mon[0] = cm20 + cm02
    Mon[1] = (cm20 - cm02) ** 2 + cm11 ** 2 * 4
    Mon[2] = (cm30 - 3 * cm12) ** 2 + (3 * cm21 - cm03) ** 2
    Mon[3] = (cm30 + cm12) ** 2 + (cm21 + cm03) ** 2
    Mon[4] = (cm30 - 3 * cm12) * (cm30 + cm12) * ((cm30 + cm12) ** 2 - 3 * (cm21 + cm03) ** 2) + (
            3 * (cm30 + cm12) ** 2 - (cm21 + cm03) ** 2)
    Mon[5] = (cm20 - cm02) * (cm30 + cm12) ** 2 - (cm21 + cm03) ** 2 + 4 * cm11 * (cm30 + cm12) * (cm21 + cm03)
    Mon[6] = (3 * cm21 - cm03) * (cm30 + cm12) * ((cm30 + cm12) ** 2 - 3 * (cm21 + cm03) ** 2) + (
            3 * cm12 - cm30) * (cm21 + cm03) * (3 * (cm30 + cm12) ** 2 - (cm21 + cm03) ** 2)
    # 8~10阶矩
    Mon[7] = (cm20 * cm02 - cm11 ** 2) / cm00 ** 4
    Mon[8] = (cm30 ** 2 * cm03 ** 2 - 6 * cm30 * cm03 * cm21 * cm12 + 4 * cm30 * cm12 ** 3) / cm00 ** 10
    Mon[9] = (cm20 * (cm21 * cm03 - cm12 ** 2) - cm11 * (cm30 * cm03 - cm21 * cm12) + cm02 * (
            cm12 * cm30 - cm21 ** 2)) / cm00 ** 7

    Mon = np.maximum(Mon, -Mon)
    Mon = np.log10(Mon)
    Mon = np.maximum(Mon, -Mon)

    # Hu_path = RESULT_FOLDER + '/SAR/Hu.csv'
    Hu_path = os.path.join(RESULT_FOLDER,'SAR','Hu.csv')
    f = open(Hu_path, 'w', newline="",encoding='utf-8-sig')
    csv_writer = csv.writer(f)
    # 构建列表头
    csv_writer.writerow(['1阶矩', '2阶矩', '3阶矩', '4阶矩', '5阶矩', '6阶矩', '7阶矩', '8阶矩', '9阶矩', '10阶矩'])
    # 写入csv文件内容
    csv_writer.writerow(Mon)
    f.close()

    return Hu_path,Mon


# if __name__ == '__main__':
#     print(MomentSeven(r'D:\back_dev_flask-master\static\result\SAR\image_f.png'))
