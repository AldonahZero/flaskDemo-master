import cv2
import math
import numpy as np
import os
from config.setting import RESULT_FOLDER


###剪切波变换
def nsst_dec(path):
    '''
    :param path: 输入图像路径
    :return: 滤波后图像存储路径
    '''
    image = cv2.imread(path)
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img_a = expm(np.float(image))
    g = np.zeros(256)
    for i in range(256):
        g[i] = math.log(i + 0.000000001)
    img_a = cv2.LUT(image, g)
    img = cv2.blur(img_a, (2, 2), borderType=cv2.BORDER_ISOLATED)
    m, n = img.shape
    img_p = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            img_p[i][j] = math.exp(img[i][j])
    image_f = np.uint8(img_p)
    # cv2.namedWindow('Filter image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Filter image', image_f)
    # image_f_path = RESULT_FOLDER + '/SAR/image_f.png'
    image_f_path = os.path.join(RESULT_FOLDER, 'SAR/image_f.png')
    cv2.imwrite(image_f_path, image_f)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image_f_path, image_f


###图像卷积处理
def cov2(img, f, strde):
    '''
    :param img: 输入图像
    :param f: 卷积窗
    :param strde: 步距
    :return:
    '''
    if img.ndim > 2:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inw, inh = img.shape
    w, h = f.shape
    outw = (inw - w) / strde + 1
    outh = (inh - h) / strde + 1
    arr = np.zeros(shape=(outw, outh))
    for g in range(outh):
        for t in range(outw):
            s = 0
            for i in range(w):
                for j in range(h):
                    s += img[i + g * strde][j + t * strde] * f[i][j]
                    # s = img[i][j] * f[i][j]
            arr[g][t] = s
    return arr


###二维Otsu算法
def otsu_2d(path):
    '''
    :param path: 输入图像存储路径
    :return: image_b_path: 二值化图像存储路径
    '''
    image = cv2.imread(path)
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    [a, b] = image.shape
    m = math.floor(a / 20)
    n = math.floor(b / 20)
    # f = np.ones([m, n]) / (m * n)
    # img_f = cov2(image, f, 1)
    img_f = cv2.blur(image, (m, n), borderType=cv2.BORDER_ISOLATED)
    # 背景补偿
    # ans = image - img_f
    ans = np.zeros([a, b])
    for i in range(a):
        for j in range(b):
            tem = float(image[i][j]) - float(img_f[i][j])
            if tem < 0:
                ans[i][j] = 0
            else:
                ans[i][j] = tem
    G = img_f
    S_xy1 = np.zeros([a, b])
    N = (ans > 0).sum()
    T = np.int32((image.sum()) / N - img_f)

    for x in range(a):
        for y in range(b):
            if float(image[x][y]) - float(img_f[x][y]) > T[x][y]:
                S_xy1[x][y] = image[x][y]
            else:
                S_xy1[x][y] = G[x][y]
    # 二维Otsu
    a0 = S_xy1
    a1 = np.zeros([a, b])
    a2 = np.empty([a, b])
    # 构造灰度均值图像
    for i in range(a):
        for j in range(b):
            for k in range(-1, 2):
                for w in range(-1, 2):
                    p = i + k
                    q = j + w
                    if p < 0 or p >= a:
                        p = i
                    if q < 0 or q >= b:
                        q = j
                    a1[i][j] = a0[p][q] + a1[i][j]
            a2[i][j] = np.uint8(a1[i][j] / 9)
    fxy = np.zeros([256, 256])
    for i in range(a):
        for j in range(b):
            c = int(a0[i][j])
            d = int(a2[i][j])
            fxy[c][d] = fxy[c][d] + 1

    pxy = fxy / (a * b)
    w0 = np.zeros([256, 256])
    ui = np.zeros([256, 256])
    uj = np.zeros([256, 256])
    w0[0][0] = pxy[0][0]
    for i in range(1, 256):
        w0[i][0] = w0[i - 1][0] + pxy[i][0]
    for i in range(1, 256):
        w0[0][i] = w0[0][i - 1] + pxy[0][i]
    for i in range(1, 256):
        for j in range(1, 256):
            w0[i][j] = w0[i - 1][j] + w0[i][j - 1] - w0[i - 1][j - 1] + pxy[i][j]
    w1 = np.ones([256, 256]) - w0
    ui[0][0] = 0
    for i in range(1, 256):
        ui[0][i] = ui[0][i - 1] + i * pxy[0][i]
    for i in range(1, 256):
        ui[i][0] = ui[i][0] + i * pxy[i][0]
    for i in range(1, 256):
        for j in range(1, 256):
            ui[i][j] = ui[i - 1][j] + ui[i][j - 1] - ui[i - 1][j - 1] + i * pxy[i][j]
    uj[0][0] = 0
    for i in range(1, 256):
        uj[0][i] = uj[0][i - 1] + i * pxy[0][i]
    for i in range(1, 256):
        uj[i][0] = uj[i - 1][0] + i * pxy[i][0]
    for i in range(1, 256):
        for j in range(1, 256):
            uj[i][j] = uj[i - 1][j] + uj[i][j - 1] - uj[i - 1][j - 1] + j * pxy[i][j]
    uti = 0  # 二维直方图总均值
    utj = 0
    for i in range(256):
        for j in range(256):
            uti = uti + i * pxy[i][j]
            utj = utj + j * pxy[i][j]
    h = np.zeros([256, 256])
    for i in range(256):
        for j in range(256):
            if w0[i][j] != 0 and w1[i][j] != 0:
                h[i][j] = ((uti * w0[i][j] - ui[i][j]) ** 2 + (utj * w0[i][j] - uj[i][j]) ** 2) / (w0[i][j] * w1[i][j])
            else:
                h[i][j] = 0
    hmax = h.max()
    for i in range(256):
        for j in range(256):
            if h[i][j] == hmax:
                s = i
                k = j
                continue
    # z = np.ones([a, b])
    z = np.zeros([a, b])
    for i in range(a):
        for j in range(b):
            if a0[i, j] > s and a2[i][j] > k:
                z[i][j] = 255
    image_b = np.uint8(z)
    # image_b_path = RESULT_FOLDER + '/SAR/image_b.png'
    image_b_path = os.path.join(RESULT_FOLDER, 'SAR/image_b.png')
    cv2.imwrite(image_b_path, image_b)
    return image_b_path, image_b

# if __name__ == '__main__':
#     path = '../../static/uploads/SAR/HB19975.JPG'
#     print(nsst_dec(path))
