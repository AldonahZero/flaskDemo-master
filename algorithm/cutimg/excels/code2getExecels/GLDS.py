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

def get_GLDS(img1):
    result = []

    img = np.array(img1).astype(np.float64)
    m, n = img.shape
    # print(img.shape)

    B = img.copy()

    C = np.zeros((m, n))

    for i in range(m - 1):
        for j in range(n - 1):
            # B[i,j]= img[i + 1, j + 1]
            a1 = float(img[i + 1, j + 1])
            a2 = float(img[i, j])
            C[i, j] = abs(round(a1 - a2))
    mn = np.min(C)
    mx = np.max(C)
    norm_C = (C - mn) * (1.0 / (mx - mn))

    hist, bins = np.histogram(norm_C.flatten(), bins=256)
    # hist_cv=cv2.calcHist([norm_C],[0],None,[256],[0,256])
    hist_re = hist / (m * n)

    MEAN = 0  # 均值
    CON = 0  # 对比度
    ASM = 0  # 角二阶矩
    ENT = 0  # 熵

    for i in range(256):
        MEAN = MEAN + (i * hist_re[i]) / 256
        CON = CON + i * i * hist_re[i]
        ASM = ASM + hist_re[i] * hist_re[i]
        if (hist_re[i] > 0):
            ENT = ENT - hist_re[i] * np.log2(hist_re[i])
    # result[0] = MEAN
    # result[1] = CON
    # result[2] = ASM
    # result[3] = ENT
    return [MEAN, CON, ASM, ENT]

def get_GLDS_excels(path_cutimg, path_excels_save):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('均值', cell_overwrite_ok=True)
    sheet2 = f.add_sheet('对比度', cell_overwrite_ok=True)
    sheet3 = f.add_sheet('角二阶矩', cell_overwrite_ok=True)
    sheet4 = f.add_sheet('熵', cell_overwrite_ok=True)

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
                    res_target = get_GLDS(gray_target)
                    sheet1.write_merge(q, q, 2 * i + 3, 2 * i + 3, res_target[0])
                    sheet2.write_merge(q, q, 2 * i + 3, 2 * i + 3, res_target[1])
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
                        res_bg = get_GLDS(gray_bg)
                        result_1 = result_1 + res_bg[0]
                        result_2 = result_2 + res_bg[1]
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

    f.save(path_excels_save +'/' + 'excel_GLDS.xls')
    return

# img_path="D:/camouflageProject/matlab2PythonCode/1.JPG"
# img1 = cv2.imread(img_path, 0)
# print(get_GLDS(img1))


# print("MEAN={}\nCON={}\nASM={}\nENT={}".format(MEAN,CON,ASM,ENT))
# cv2.imshow("pandas",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# get_GLDS_excels('D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM',
#                 'D:/Python/Python/WZ_GLDM/webNew2/static/excels_save_3.21')