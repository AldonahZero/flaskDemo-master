import os
import math
import numpy as np
import csv
from .process_pre import nsst_dec
RESULT_FOLDER = os.path.join('algorithm', 'SAR', 'result')


def circshift(v, shift, num):  # 循环右移
    '''
    :param v: 待移数值
    :param shift: 移动位数
    :param num: 数值总位数
    :return:
    '''
    shift &= (num - 1)
    if shift == 0:
        return v
    return ((v >> shift) + (v & 1) * (2 ** (num - 1)))


def getCircularLBPFeature(path, radius=3, neighbors=8):
    '''
    :param img: 输入图像
    :param radius: 圆形半径
    :param neighbors: 像素点邻域点数
    :return:
    '''
    _, img = nsst_dec(path)
    rows, cols = img.shape
    rows = np.int16(rows)  # int16:有符号，两字节，正数向无穷大方向取整，负数从负无穷方向取整
    cols = np.int16(cols)
    imglbp = np.uint8(np.zeros([rows - 2 * radius, cols - 2 * radius]))  # 图像边界无领域像素点
    for n in range(neighbors):
        # 计算采样点对于中心坐标的偏移量rx，ry
        rx = radius * np.cos(2.0 * math.pi * n / neighbors)
        ry = -1 * radius * np.sin(2.0 * math.pi * n / neighbors)
        # 对采样点偏移量分别上下取整
        x1 = np.floor(rx)
        x2 = np.ceil(rx)
        y1 = np.floor(ry)
        y2 = np.ceil(ry)
        # 将坐标偏移量映射到0-1之间
        tx = rx - x1
        ty = ry - y1
        # 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间差值有关
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = img[i][j]
                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbor = img[i + int(x1)][j + int(y1)] * w1 + img[i + int(x1)][j + int(y2)] * w2 + img[i + int(x2)][
                    j + int(y1)] * w3 + img[i + int(x2)][j + int(y2)] * w4
                # LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应LBP值通过移位取得
                if neighbor > center:
                    flag = 1
                else:
                    flag = 0
                a = flag << (neighbors - n - 1)
                k = imglbp[i - radius][j - radius] | a
                imglbp[i - radius][j - radius] = k
                # imglbp[i - radius + 1][j - radius + 1] = abs(imglbp[i - radius + 1][j - radius + 1])
    # 旋转不变
    for i in range(rows - 2 * radius):
        for j in range(cols - 2 * radius):
            if imglbp[i][j] < 0:
                currentValue = 0
            else:
                currentValue = imglbp[i][j]
            # currentValue = abs(imglbp[i][j])
            minValue = currentValue
            # currentValue = bin(currentValue)
            # 循环右移
            for n in range(1, neighbors + 1):
                current = currentValue
                current = circshift(current, 1, neighbors)
                if current < minValue:
                    minValue = current
            imglbp[i][j] = minValue
    # 对LBP特征进行显示
    lbp_stat = np.zeros([1, 256])
    num1, num2 = imglbp.shape
    for i in range(num1):
        for j in range(num2):
            k = imglbp[i][j]
            lbp_stat[0][k] += 1
    axis = []
    for i in range(256):
        axis.append(i)
    lbp = []
    for i in range(256):
        lbp.append(lbp_stat[0][i])
    # LBP_features_path = RESULT_FOLDER + '/SAR/LBP.csv'
    LBP_features_path = os.path.join(RESULT_FOLDER, 'SAR','LBP.csv')
    f = open(LBP_features_path, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    # 构建列表头
    csv_writer.writerow(axis)
    # 写入csv文件内容
    csv_writer.writerow(lbp)
    f.close()
    # lbp_stat = lbp_stat.T
    # plt.figure()
    # plt.plot(axis, lbp_stat)
    # LBP_image_path = RESULT_FOLDER + '/SAR/LBP.png'
    # plt.savefig(LBP_image_path)
    # # plt.show()
    return LBP_features_path, lbp

# if __name__ == '__main__':
#     print(getCircularLBPFeature(r'D:\back_dev_flask-master\static\result\SAR\image_f.png'))
