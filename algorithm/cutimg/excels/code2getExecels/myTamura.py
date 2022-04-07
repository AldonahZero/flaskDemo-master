import cv2
import numpy as np
from scipy import signal
import math
from scipy import signal as sg
import cv2
import numpy as np
import os
import xlwt
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

# 设置表格样式


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def coarseness(image, kmax):
    image = np.array(image)
    print(" ")
    h = image.shape[0]
    w = image.shape[1]
    kmax = kmax if (np.power(2, kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2, kmax) < h) else int(np.log(h) / np.log(2))
    average_gray = np.zeros([kmax, w, h])
    A = np.zeros([h,w,2**kmax])

    for i in range(2**(kmax-1) - 1,h-2**(kmax-1)):
        for j in range(2**(kmax-1) - 1,w-2**(kmax-1)):
            for k in range(1,kmax+1):
                # a=i - 2 ** (k - 1)
                # b=i + 2 ** (k - 1) - 1
                # c=j - 2 ** (k - 1)
                # d=j + 2 ** (k - 1) - 1
                A[i, j, k] = np.mean(image[i - 2 ** (k - 1)-1: i + 2 **(k - 1)-1, j - 2 **(k - 1)-1: j + 2 ** (k - 1)-1])



    Eh=np.ones((h-2**(kmax-1),w-2**(kmax-1),kmax))
    Ev=np.ones((h-2**(kmax-1),w-2**(kmax-1),kmax))

    for i in range(2**(kmax-1) - 1,h-2**(kmax-1)):
        for j in range(2**(kmax-1) - 1,w-2**(kmax-1)):
            for k in range(1,kmax+1):
                inter_1=i+2**(k-1)
                inter_2=i-2**(k-1)
                Eh[i, j, k-1] =abs(A[inter_1, j, k-1]- A[inter_2,j,k-1])
                Ev[i,j,k-1] =abs(A[i,j+2**(k-1),k-1]-A[i,j-2**(k-1),k-1])
    Sbest=np.ones((h-2**(kmax-1),w-2**(kmax-1)))

    su_1=np.sum(Eh)
    su_2=np.sum(Ev)
    for i in range(2**(kmax-1)-1,h-2**(kmax-1)):
        for j in range(2**(kmax-1)-1,w-2**(kmax-1)):
            maxEh=max(Eh[i,j,:])
            p=list(Eh[i,j,:]).index(max(Eh[i,j,:]))
            maxEv=max(Ev[i,j,:])
            q = list(Ev[i, j, :]).index(max(Ev[i, j, :]))
            if (maxEh>maxEv):
                maxkk=p
            else:
                maxkk=q
            Sbest[i,j]=2**(maxkk + 1)
    # sum = np.sum(Sbest)
    Fcrs = np.mean(Sbest)
    return Fcrs




def contrast(image):
    image = np.array(image)
    image = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    m4 = np.mean(np.power(image - np.mean(image), 4))
    v = np.var(image)

    alfa4 = m4 / np.power(v, 2)
    delta = np.std(image,1)
    fcon = delta/ np.power(alfa4, 0.25)
    return fcon


def directionality(gray):
    # w, h = gray.shape
    h, w = gray.shape
    print(' ')

    GradientH = [[-1, 0, 1],
                 [-1, 0, 1],
                 [-1, 0, 1]]
    GradientV = [[1, 1, 1],
                 [0, 0, 0],
                 [-1, -1, -1]]

    MHconv = sg.convolve2d(gray, GradientH)
    MH = MHconv[2:h, 2:w]

    MVconv = sg.convolve2d(gray, GradientV)
    MV = MVconv[2:h, 2:w]

    MG = (abs(MH) + abs(MV)) / 2

    validH = h - 2
    validW = w - 2
    inter = np.zeros((validH, validW))

    for i in range(0, validH):
        for j in range(0, validW):
            inter[i, j] = math.atan(MV[i, j] / MH[i, j]) + (np.pi / 2)

    n = 16
    t = 12
    Nsita = np.zeros((1, n))
    Nsita.resize(n)

    for i in range(0, validH):
        for j in range(0, validW):
            for k in range(0, n):
                if inter[i, j] >= (2 * (k - 1) * np.pi / 2 / n) and inter[i, j] < (
                        (2 * (k - 1) + 1) * np.pi / 2 / n) and MG[i, j] >= t:
                    Nsita[k] = Nsita[k] + 1
    HD = np.zeros(n)
    for k in range(0, n):
        HD[k] = Nsita[k] / sum(Nsita[:])
    HD = list(HD)
    FIp = HD.index(max(HD))

    Fdir = 0
    for k in range(0, n):
        Fdir = Fdir + (k - FIp) ** 2 * HD[k]
    return Fdir,inter



def linelikeness(gray,sita,d):
    # 构建方向共生矩阵
    d = d  # d为共生矩阵计算时的像素间隔距离
    n=16
    # w, h = gray.shape
    h, w = gray.shape
    # 构建方向共生矩阵
    d = 4  # d为共生矩阵计算时的像素间隔距离
    PDd1 = np.zeros((n, n))
    PDd2 = np.zeros((n, n))
    PDd3 = np.zeros((n, n))
    PDd4 = np.zeros((n, n))
    PDd5 = np.zeros((n, n))
    PDd6 = np.zeros((n, n))
    PDd7 = np.zeros((n, n))
    PDd8 = np.zeros((n, n))
    for i in range(d + 1, h - d - 2 - 5):  # 本来只-2，但是会导致后面sita[i+ d, j]溢出，必须多-5
        for j in range(d + 1, w - d - 2 - 5):  # 本来只-2，但是会导致后面sita[i- d, j+d]溢出，必须多-5
            for m1 in range(0, n):
                for m2 in range(0, n):
                    # 下方向
                    if (sita[i, j] >= (2 * (m1 - 1) * np.pi / 2 / n) and sita[i, j] < (
                            (2 * (m1 - 1) + 1) * np.pi / 2 / n)) and (
                            sita[i + d, j] >= (2 * (m2 - 1) * np.pi / 2 / n) and sita[i + d, j] < (
                            (2 * (m2 - 1) + 1) * np.pi / 2 / n)):
                        PDd1[m1, m2] = PDd1[m1, m2] + 1
                    # 上方向
                    if (sita[i, j] >= (2 * (m1 - 1) * np.pi / 2 / n) and sita[i, j] < (
                            (2 * (m1 - 1) + 1) * np.pi / 2 / n)) and (
                            sita[i - d, j] >= (2 * (m2 - 1) * np.pi / 2 / n) and sita[i - d, j] < (
                            (2 * (m2 - 1) + 1) * np.pi / 2 / n)):
                        PDd2[m1, m2] = PDd1[m1, m2] + 1
                    # 右方向
                    if (sita[i, j] >= (2 * (m1 - 1) * np.pi / 2 / n) and sita[i, j] < (
                            (2 * (m1 - 1) + 1) * np.pi / 2 / n)) and (
                            sita[i, j + d] >= (2 * (m2 - 1) * np.pi / 2 / n) and sita[i, j + d] < (
                            (2 * (m2 - 1) + 1) * np.pi / 2 / n)):
                        PDd3[m1, m2] = PDd1[m1, m2] + 1
                    # 左方向
                    if (sita[i, j] >= (2 * (m1 - 1) * np.pi / 2 / n) and sita[i, j] < (
                            (2 * (m1 - 1) + 1) * np.pi / 2 / n)) and (
                            sita[i, j - d] >= (2 * (m2 - 1) * np.pi / 2 / n) and sita[i, j - d] < (
                            (2 * (m2 - 1) + 1) * np.pi / 2 / n)):
                        PDd4[m1, m2] = PDd1[m1, m2] + 1
                    # 右下方向
                    if (sita[i, j] >= (2 * (m1 - 1) * np.pi / 2 / n) and sita[i, j] < (
                            (2 * (m1 - 1) + 1) * np.pi / 2 / n)) and (
                            sita[i + d, j + d] >= (2 * (m2 - 1) * np.pi / 2 / n) and sita[i + d, j + d] < (
                            (2 * (m2 - 1) + 1) * np.pi / 2 / n)):
                        PDd5[m1, m2] = PDd1[m1, m2] + 1
                    # 右上方向
                    if (sita[i, j] >= (2 * (m1 - 1) * np.pi / 2 / n) and sita[i, j] < (
                            (2 * (m1 - 1) + 1) * np.pi / 2 / n)) and (
                            sita[i - d, j + d] >= (2 * (m2 - 1) * np.pi / 2 / n) and sita[i - d, j + d] < (
                            (2 * (m2 - 1) + 1) * np.pi / 2 / n)):
                        PDd6[m1, m2] = PDd1[m1, m2] + 1
                    # 左下方向
                    if (sita[i, j] >= (2 * (m1 - 1) * np.pi / 2 / n) and sita[i, j] < (
                            (2 * (m1 - 1) + 1) * np.pi / 2 / n)) and (
                            sita[i + d, j - d] >= (2 * (m2 - 1) * np.pi / 2 / n) and sita[i + d, j - d] < (
                            (2 * (m2 - 1) + 1) * np.pi / 2 / n)):
                        PDd7[m1, m2] = PDd1[m1, m2] + 1
                    # 左上方向
                    if (sita[i, j] >= (2 * (m1 - 1) * np.pi / 2 / n) and sita[i, j] < (
                            (2 * (m1 - 1) + 1) * np.pi / 2 / n)) and (
                            sita[i - d, j - d] >= (2 * (m2 - 1) * np.pi / 2 / n) and sita[i - d, j - d] < (
                            (2 * (m2 - 1) + 1) * np.pi / 2 / n)):
                        PDd8[m1, m2] = PDd1[m1, m2] + 1



    f = np.zeros(8)
    g = np.zeros(8)
    for i in range(1, n ):
        for j in range(1, n ):
            f[0] = f[0] +PDd1[i, j] * math.cos((i - j) * 2 * np.pi / n)
            g[0] = g[0] + PDd1[i, j]
            f[1] = f[1]  +PDd2[i, j] * math.cos((i - j) * 2 * np.pi / n)
            g[1] = g[1] + PDd2[i, j]
            f[2] = f[2] +PDd3[i, j] * math.cos((i - j) * 2 * np.pi / n)
            g[2] = g[2] + PDd3[i, j]
            f[3] = f[3] +PDd4[i, j] * math.cos((i - j) * 2 * np.pi / n)
            g[3] = g[3] + PDd4[i, j]
            f[4] = f[4] +PDd5[i, j] * math.cos((i - j) * 2 * np.pi / n)
            g[4] = g[4] + PDd5[i, j]
            f[5] = f[5] +PDd6[i, j] * math.cos((i - j) * 2 * np.pi / n)
            g[5] = g[5] + PDd6[i, j]
            f[6] = f[6] +PDd7[i, j] * math.cos((i - j) * 2 * np.pi / n)
            g[6] = g[6] + PDd7[i, j]
            f[7] = f[7] +PDd8[i, j] * math.cos((i - j) * 2 * np.pi / n)
            g[7] = g[7] + PDd8[i, j]

    tempM = f / g
    Flin = np.max(tempM)  # 取8个方向的线性度最大值作为图片的线性度
    return Flin


def get_tamura(img):
    fcrs = coarseness(img, 5)
    # print("coarseness: %f" % fcrs)
    # print(fcrs)
    fcon = contrast(img)
    # print("contrast: %f" % fcon)
    fdir,sita= directionality(img)
    # print("directionality: %f" % fdir)
    flin = linelikeness(img,sita,4)
    # print("linelikeness: %f" % flin)
    return [np.round(fcrs, 2), np.round(fcon, 2), np.round(fdir, 2), np.round(flin, 2)]


def get_Tamura_excels(path_cutimg, path_excels_save):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('粗糙度', cell_overwrite_ok=True)
    sheet2 = f.add_sheet('对比度', cell_overwrite_ok=True)
    sheet3 = f.add_sheet('方向度', cell_overwrite_ok=True)
    sheet4 = f.add_sheet('线性度', cell_overwrite_ok=True)

    row0 = ["c/nc", "v/i", "h",
            "result1_0°_t", "result1_0°_b", "result2_0°_t", "result2_0°_b",
            "result3_0°_t", "result3_0°_b", "result4_0°_t", "result4_0°_b",
            "result5_0°_t", "result5_0°_b", "result6_0°_t", "result6_0°_b",
            "result7_0°_t", "result7_0°_b", "result8_0°_t", "result8_0°_b",
            "result9_0°_t", "result9_0°_b", "result10_0°_t", "result10_0°_b"
            ]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet2.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet3.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet4.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    # 修改灰度直方图的方向
    angle = 2  # 0-0° 1-90° 2-180° 3-270°

    q = 0
    path = path_cutimg
    for filename1 in os.listdir(path):
        path1 = path + '/' + filename1
        for filename2 in os.listdir(path1):
            path2 = path1 + '/' + filename2
            for filename3 in os.listdir(path2):
                path3 = path2 + '/' + filename3

                q = q + 1
                sheet1.write_merge(q, q, 0, 0, filename1)
                sheet1.write_merge(q, q, 1, 1, filename2)
                sheet1.write_merge(q, q, 2, 2, filename3)

                sheet2.write_merge(q, q, 0, 0, filename1)
                sheet2.write_merge(q, q, 1, 1, filename2)
                sheet2.write_merge(q, q, 2, 2, filename3)

                sheet3.write_merge(q, q, 0, 0, filename1)
                sheet3.write_merge(q, q, 1, 1, filename2)
                sheet3.write_merge(q, q, 2, 2, filename3)

                sheet4.write_merge(q, q, 0, 0, filename1)
                sheet4.write_merge(q, q, 1, 1, filename2)
                sheet4.write_merge(q, q, 2, 2, filename3)


                for i in range(10):
                    # print(i)
                    target_name = str(i) + str(4) + '.jpg'
                    img_target = cv2.imread(path3 + '/' + target_name)

                    if img_target is None:
                        # print(i)
                        sheet1.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet2.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet3.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet4.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')

                        continue
                    # 计算目标图像的灰度共生矩阵值
                    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
                    gray_target = img_as_ubyte(gray_target)
                    # inds = np.digitize(gray_target, bins)
                    # max_value = inds.max() + 1
                    # matrix_target = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                    #                              normed=False, symmetric=False)
                    res_target = get_tamura(gray_target)
                    sheet1.write_merge(q, q, 2 * i + 3, 2 * i + 3, res_target[0])
                    sheet2.write_merge(q, q, 2 * i + 3, 2 * i + 3, res_target[1][0])
                    sheet3.write_merge(q, q, 2 * i + 3, 2 * i + 3, res_target[2])
                    sheet4.write_merge(q, q, 2 * i + 3, 2 * i + 3, res_target[3])
                    ##########################

                    [a1, b1, c1] = img_target.shape

                    result_1 = 0
                    result_2 = 0
                    result_3 = 0
                    result_4 = 0

                    cnt = 0
                    sign = True
                    for j in range(9):
                        if j == 4:
                            continue
                        backg_name = str(i) + str(j) + '.jpg'
                        # print(backg_name)
                        if not os.path.exists(path3 + '/' + backg_name):
                            continue
                        img_backg = cv2.imread(path3 + '/' + backg_name)
                        if img_backg is None:
                            continue
                        [a2, b2, c2] = img_backg.shape
                        # print(backg_name)
                        a_min = min(a1, a2)
                        b_min = min(b1, b2)
                        img_target2 = img_target[0:a_min, 0:b_min, :]
                        img_backg2 = img_backg[0:a_min, 0:b_min, :]
                        # 修改函数#########################################################################
                        gray_bg = cv2.cvtColor(img_backg2, cv2.COLOR_BGR2GRAY)
                        gray_bg = img_as_ubyte(gray_bg)
                        # inds2 = np.digitize(gray_bg, bins)
                        # max_value2 = inds2.max() + 1
                        # matrix_bg = greycomatrix(inds2, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        #                          levels=max_value2,
                        #                          normed=False, symmetric=False)
                        res_bg = get_tamura(gray_bg)
                        result_1 = result_1 + res_bg[0]
                        result_2 = result_2 + res_bg[1][0]
                        result_3 = result_3 + res_bg[2]
                        result_4 = result_4 + res_bg[3]
                        # 以上为修改函数####################################################################
                        cnt = cnt + 1
                        sign = False
                    if sign:
                        everage_1 = 'NA'
                        everage_2 = 'NA'
                        everage_3 = 'NA'
                        everage_4 = 'NA'
                    else:
                        everage_1 = result_1 / cnt
                        everage_2 = result_2 / cnt
                        everage_3 = result_3 / cnt
                        everage_4 = result_4 / cnt
                    sheet1.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_1)
                    sheet2.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_2)
                    sheet3.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_3)
                    sheet4.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_4)
                    # print('end')

    f.save(path_excels_save +'/' + 'excel_Tamura.xls')
    return

# get_Tamura_excels('D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM',
#                 'D:/Python/Python/WZ_GLDM/webNew2/static/excels_save_3.21')

# if __name__ == '__main__':
#
#     img = cv2.imread('D:/camouflageProject/matlab2PythonCode/1.JPG', cv2.IMREAD_GRAYSCALE)
#     fcrs = coarseness(img, 5)
#     print("coarseness: %f" % fcrs)
#     fcon = contrast(img)
#     print("contrast: %f" % fcon)
#     fdir,sita= directionality(img)
#     print("directionality: %f" % fdir)
#     flin = linelikeness(img,sita,4)
#     print("linelikeness: %f" % flin)
