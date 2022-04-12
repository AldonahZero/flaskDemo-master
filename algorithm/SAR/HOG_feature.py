import os
import math
import csv
import numpy as np
from .process_pre import nsst_dec
from config.setting import RESULT_FOLDER


def gamma_cor():  # gamma校正
    cor = []
    for i in range(0, 256):
        a = (((i + 0.5) / 255) ** (0.454545)) * 255 - 0.5
        cor.append(a)
    return cor


def HOG_feature(path):
    '''
    :param img: 输入图像
    :return:
    '''
    _, img1 = nsst_dec(path)
    [m, n] = img1.shape
    img_c = np.zeros([m, n])
    cor = np.uint8(gamma_cor())
    # gamma校正
    for i in range(m):
        for j in range(n):
            img_c[i, j] = cor[img1[i, j]]
    [Iy, Ix] = np.gradient(img_c)
    Gra_mag = (Ix ** 2 + Iy ** 2) ** 0.5  # 边缘强度
    Gra_dir = Iy / Ix  # 边缘斜率

    # 求cell
    step = 4  # step*step个像素作为一个单元
    orient = 18  # 方向直方图的方向个数
    jiao = 360 / orient  # 每个方向包含的角度数
    x1 = math.floor(m / step)  # 横向单元格
    y1 = math.floor(n / step)  # 纵向单元格
    Cell = np.zeros((x1, y1, 1, 9), dtype=object)
    ii = 0
    for i in range(0, x1 * step, step):
        jj = 0
        for j in range(0, y1 * step, step):
            tmpx = Ix[i:i + step, j:j + step];
            tmped = Gra_mag[i:i + step, j:j + step]
            if tmped.min() != 0:
                tmped = tmped / tmped.sum()
            tmpphase = Gra_dir[i:i + step, j:j + step]
            Hist1 = np.zeros([1, orient])
            Hist = np.zeros([1, 9])
            for p in range(step):
                for q in range(step):
                    if np.isnan(tmpphase[p, q]) == 1:
                        tmpphase[p, q] = 0
                    ang = math.atan(tmpphase[p, q])
                    ang = (ang * 180 / math.pi) % 360
                    if tmpx[p, q] < 0:
                        if (ang < 90) == 1:
                            ang = ang + 180
                        else:
                            if (ang > 270) == 1:
                                ang = ang - 180
                    ang = ang + 0.0000001
                    ak = ang / jiao
                    if ak < 1 or ak > 17:
                        Hist1[0][int(ak)] = tmped[p, q] + Hist1[0][int(ak)]
                    else:
                        if round(ak) > ak:
                            Hist1[0][round(ak)] = Hist1[0][round(ak)] + (ak - int(ak)) * tmped[p, q]
                            Hist1[0][round(ak) - 1] = Hist1[0][round(ak) - 1] + (round(ak) - ak) * tmped[p, q]
                        else:
                            Hist1[0][round(ak)] = Hist1[0][round(ak)] + (math.ceil(ak) - ak) * tmped[p, q]
                            Hist1[0][round(ak) + 1] = Hist1[0][round(ak + 1)] + (ak - int(ak)) * tmped[p, q]
            for w in range(9):
                Hist[0][w] = Hist1[0][w] + Hist1[0][w + 9]
            Cell[ii][jj] = Hist
            jj = jj + 1
        ii = ii + 1

    # 求feature，2*2个cell合成一个block
    [m1, n1, m2, n2] = Cell.shape
    feature = np.zeros([1, (m1 - 1) * (n1 - 1), 1, 36]);
    for i in range(m1 - 1):
        for j in range(n1 - 1):
            f = np.array([Cell[i][j], Cell[i][j + 1], Cell[i + 1][j], Cell[i + 1][j + 1]])
            f = np.resize(f, (1, 36))
            f = f / ((f ** 2).sum() + 0.000001)
            feature[0][i * (n1 - 1) + j] = f

    # 将特征进行显示
    lon = feature.shape
    f = np.reshape(feature, (lon[1], 36))
    feature_h = np.zeros([1, 36])
    for i in range(lon[1]):
        feature_h = feature_h + f[i]

    # hog特征显示
    h = []
    for i in range(36):
        h.append(feature_h[0][i])
    # hog_features_path = RESULT_FOLDER + '/SAR/hog.csv'
    hog_features_path = os.path.join(RESULT_FOLDER, 'SAR/hog.csv')
    f = open(hog_features_path, 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    # 构建列表头
    csv_writer.writerow(
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
         '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35'])
    # 写入csv文件内容
    csv_writer.writerow(h)
    f.close()
    # x = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    #      '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35')
    # plt.figure()
    # plt.bar(x, h)
    # HOG_image_path = RESULT_FOLDER + '/SAR/HOG.jpg'
    # plt.savefig(HOG_image_path)
    # # plt.show()

    return hog_features_path, h

# if __name__ == '__main__':
#     print(HOG_feature(r'D:\back_dev_flask-master\static\result\SAR\image_f.png'))
