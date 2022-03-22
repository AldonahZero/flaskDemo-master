import os
import cv2
import xlwt
import matplotlib.pyplot as plt
import math
import numpy as np


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def plt_hist(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def hist_square(hist_1, hist_2):
    sum1 = 0
    sum2 = 0
    ex = 0
    ey = 0
    exy = 0
    dx = 0
    dy = 0
    cov = 0
    skewness = 0  # 目标图像的偏斜度，代表目标图像分布的位置
    # 协方差系数
    p = 0
    for i in range(0, 255):
        sum1 += hist_1[i]
        sum2 += hist_2[i]
    # 求灰度直方图的均值（期望）
    ex = sum1 / 255
    ey = sum2 / 255
    # 求平方的期望
    for i in range(0, 255):
        dx += pow(hist_1[i], 2)
        dy += pow(hist_2[i], 2)
    dx = dx / 255
    dy = dy / 255
    # 求方差=平方的期望减期望的平方
    dx = dx - pow(ex, 2)
    dy = dy - pow(ey, 2)
    # 求协方差cov=E[XY]-E[X]E[Y]
    for i in range(0, 255):
        exy += hist_1[i]*hist_2[i]
    exy = exy / 255
    cov = exy - ex*ey
    # 求协方差系数 p = cov / (dx开平方*dy开平方)
    p = cov / (pow(dx, 0.5)*pow(dy, 0.5))
    return ex, ey, exy, dx, dy, cov, p, skewness


'''
f = xlwt.Workbook()
sheet1 = f.add_sheet('feature', cell_overwrite_ok=True)
row0 = ["c/nc", "v/i", "h",
        "result1_协方差",  "result1_协方差系数", "result2_协方差", "result2_协方差系数",
        "result3_协方差",  "result3_协方差系数", "result4_协方差", "result4_协方差系数",
        "result5_协方差",  "result5_协方差系数", "result6_协方差", "result6_协方差系数",
        "result7_协方差",  "result7_协方差系数", "result8_协方差", "result8_协方差系数",
        "result9_协方差",  "result9_协方差系数", "result10_协方差", "result10_协方差系数"
        ]
for i in range(0, len(row0)):
    sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

# 批处理
q = 0
path = 'D:/chuzhou_data_original' # 原始图像
path_bitwise = 'D:/chuzhou_data_bitwise' # 掩膜
for filename1 in os.listdir(path):
    path1 = path + '/' + filename1
    path1_bitwise = path_bitwise + '/' + filename1
    for filename2 in os.listdir(path1):
        path2 = path1 + '/' + filename2
        path2_bitwise = path1_bitwise + '/' + filename2
        for filename3 in os.listdir(path2):
            path3 = path2 + '/' + filename3
            path3_bitwise = path2_bitwise + '/' + 'bitwise' + filename3

            q = q + 1
            sheet1.write_merge(q, q, 0, 0, filename1)
            sheet1.write_merge(q, q, 1, 1, filename2)
            sheet1.write_merge(q, q, 2, 2, filename3)
            img_name = 0

            # 用list代替数组(最快)
            for original_name in os.listdir(path3):
                array_t_length = 0
                array_b_length = 0

                # 如果原图为空
                if not os.path.exists(path3 + '/' + original_name):
                    img_name = img_name + 1
                    continue
                if not os.path.exists(path3_bitwise + '/' + str(img_name) + '.jpg'):
                    img_name = img_name + 1
                    continue

                img_original = cv2.imread(path3 + '/' + original_name)
                bitwise = cv2.imread(path3_bitwise + '/' + str(img_name) + '.jpg')
                bitwise = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)
                # print('path of o', path3 + '/' + original_name)
                gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
                print(bitwise.shape)
                print(gray_original.shape)
                [height, width] = bitwise.shape
                for i in range(height):
                    for j in range(width):
                        pixel_value2 = bitwise[i, j]
                        if pixel_value2 == 0:
                            array_b_length = array_b_length + 1
                        else:
                            array_t_length = array_t_length + 1
                array_t = [0]*array_t_length
                array_b = [0]*array_b_length
                print(len(array_t), len(array_b))
                t = 0
                b = 0
                for i in range(height):
                    for j in range(width):
                        pixel_value = gray_original[i, j]
                        pixel_value2 = bitwise[i, j]
                        if pixel_value2 == 0:
                            array_b[b] = pixel_value
                            # print('b = ', b)
                            b = b + 1

                        else:
                            # print('t = ', t)
                            array_t[t] = pixel_value
                            t = t + 1
                array_t = np.array(array_t, np.float32)
                array_b = np.array(array_b, np.float32)
                hist_t = cv2.calcHist([array_t], [0], None, [256], [0, 255])
                hist_b = cv2.calcHist([array_b], [0], None, [256], [0, 255])

                ex1, ey1, exy1, dx1, dy1, cov1, p1, skewness1 = hist_square(hist_t, hist_b)
                cov1 = "%.2f" % cov1
                p1 = "%.2f" % p1
                sheet1.write_merge(q, q, 2 * img_name + 3, 2 * img_name + 3, cov1)
                sheet1.write_merge(q, q, 2 * img_name + 4, 2 * img_name + 4, p1)
                print('cov1, p1 = ', cov1, p1)

                img_name = img_name + 1

f.save('D:/chuzhou_data_excel_save/gray_histogram_test.xls')
'''

# 单张图像测试

def gray_hist_main(img_original, bitwise):
    array_t_length = 0
    array_b_length = 0

    bitwise = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)
    # print('path of o', path3 + '/' + original_name)
    gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    # print(bitwise.shape)
    # print(gray_original.shape)
    [height, width] = bitwise.shape
    for i in range(height):
        for j in range(width):
            pixel_value2 = bitwise[i, j]
            if pixel_value2 == 0:
                array_b_length = array_b_length + 1
            else:
                array_t_length = array_t_length + 1
    array_t = [0] * array_t_length
    array_b = [0] * array_b_length
    # print(len(array_t), len(array_b))
    t = 0
    b = 0
    for i in range(height):
        for j in range(width):
            pixel_value = gray_original[i, j]
            pixel_value2 = bitwise[i, j]
            if pixel_value2 == 0:
                array_b[b] = pixel_value
                # print('b = ', b)
                b = b + 1

            else:
                # print('t = ', t)
                array_t[t] = pixel_value
                t = t + 1
    array_t = np.array(array_t, np.float32)
    array_b = np.array(array_b, np.float32)
    hist_t = cv2.calcHist([array_t], [0], None, [256], [0, 255])
    hist_b = cv2.calcHist([array_b], [0], None, [256], [0, 255])
    plt.figure(1)
    plt.hist(hist_t, 256, [0, 256])
    plt.figure(2)
    plt.hist(hist_b, 256, [0, 256])
    # plt.show()
    ex1, ey1, exy1, dx1, dy1, cov1, p1, skewness1 = hist_square(hist_t, hist_b)
    cov1 = "%.2f" % cov1
    p1 = "%.2f" % p1
    print('p1 = ', p1)

    return p1


# 批处理获得表
def get_gray_histogram_excels(path_original_img, path_bitwise, path_excels_save):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('feature', cell_overwrite_ok=True)
    row0 = ["c/nc", "v/i", "h",
            "result1_协方差", "result1_协方差系数", "result2_协方差", "result2_协方差系数",
            "result3_协方差", "result3_协方差系数", "result4_协方差", "result4_协方差系数",
            "result5_协方差", "result5_协方差系数", "result6_协方差", "result6_协方差系数",
            "result7_协方差", "result7_协方差系数", "result8_协方差", "result8_协方差系数",
            "result9_协方差", "result9_协方差系数", "result10_协方差", "result10_协方差系数"
            ]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    # 批处理
    q = 0
    # path = 'D:/chuzhou_data_original'  # 原始图像
    path = path_original_img
    # path_bitwise = 'D:/chuzhou_data_bitwise'  # 掩膜
    path_bitwise = path_bitwise
    for filename1 in os.listdir(path):
        path1 = path + '/' + filename1
        path1_bitwise = path_bitwise + '/' + filename1
        for filename2 in os.listdir(path1):
            path2 = path1 + '/' + filename2
            path2_bitwise = path1_bitwise + '/' + filename2
            for filename3 in os.listdir(path2):
                path3 = path2 + '/' + filename3
                path3_bitwise = path2_bitwise + '/' + filename3
                # print(path3)
                # print(path3_bitwise)

                q = q + 1
                sheet1.write_merge(q, q, 0, 0, filename1)
                sheet1.write_merge(q, q, 1, 1, filename2)
                sheet1.write_merge(q, q, 2, 2, filename3)
                img_name = 0

                # 用list代替数组(最快)
                for original_name in os.listdir(path3):
                    array_t_length = 0
                    array_b_length = 0

                    # 如果原图为空
                    if not os.path.exists(path3 + '/' + original_name):
                        img_name = img_name + 1
                        # print(1)
                        continue
                    if not os.path.exists(path3_bitwise + '/' + original_name):
                        img_name = img_name + 1
                        # print(img_name)
                        continue

                    img_original = cv2.imread(path3 + '/' + original_name)
                    bitwise = cv2.imread(path3_bitwise + '/' + original_name)
                    bitwise = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)
                    # print('path of o', path3 + '/' + original_name)
                    gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
                    print(bitwise.shape)
                    print(gray_original.shape)
                    [height, width] = bitwise.shape
                    for i in range(height):
                        for j in range(width):
                            pixel_value2 = bitwise[i, j]
                            if pixel_value2 == 0:
                                array_b_length = array_b_length + 1
                            else:
                                array_t_length = array_t_length + 1
                    array_t = [0] * array_t_length
                    array_b = [0] * array_b_length
                    print(len(array_t), len(array_b))
                    t = 0
                    b = 0
                    for i in range(height):
                        for j in range(width):
                            pixel_value = gray_original[i, j]
                            pixel_value2 = bitwise[i, j]
                            if pixel_value2 == 0:
                                array_b[b] = pixel_value
                                # print('b = ', b)
                                b = b + 1

                            else:
                                # print('t = ', t)
                                array_t[t] = pixel_value
                                t = t + 1
                    array_t = np.array(array_t, np.float32)
                    array_b = np.array(array_b, np.float32)
                    hist_t = cv2.calcHist([array_t], [0], None, [256], [0, 255])
                    hist_b = cv2.calcHist([array_b], [0], None, [256], [0, 255])

                    ex1, ey1, exy1, dx1, dy1, cov1, p1, skewness1 = hist_square(hist_t, hist_b)
                    cov1 = "%.2f" % cov1
                    p1 = "%.2f" % p1
                    sheet1.write_merge(q, q, 2 * img_name + 3, 2 * img_name + 3, cov1)
                    sheet1.write_merge(q, q, 2 * img_name + 4, 2 * img_name + 4, p1)
                    # print('cov1, p1 = ', cov1, p1)

                    img_name = img_name + 1

    f.save(path_excels_save + '/' + 'excel_gray_histogram.xls')
    return




# # 调用测试
# get_gray_histogram_excels('D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM_original',
#                           'D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM_bitwise',
#                           'D:/Python/Python/WZ_GLDM/webNew2/static/excels_save_3.21')


