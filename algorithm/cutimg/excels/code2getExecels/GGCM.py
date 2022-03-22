import cv2
import numpy as np
# from numba import jit
np.set_printoptions(suppress=True)  # 输出时禁止科学表示法，直接输出小数值
import numpy as np
import cv2
import os
import xlwt
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

def glgcm(img_gray, ngrad=16, ngray=16):
    '''Gray Level-Gradient Co-occurrence Matrix,取归一化后的灰度值、梯度值分别为16、16'''
    # 利用sobel算子分别计算x-y方向上的梯度值
    gsx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    height, width = img_gray.shape
    grad = (gsx ** 2 + gsy ** 2) ** 0.5 # 计算梯度值
    grad = np.asarray(1.0 * grad * (ngrad-1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * img_gray * (ngray-1) / img_gray.max(), dtype=np.int16) # 0-255变换为0-15
    gray_grad = np.zeros([ngray, ngrad]) # 灰度梯度共生矩阵
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad = 1.0 * gray_grad / (height * width) # 归一化灰度梯度矩阵，减少计算量
    glgcm_features = get_glgcm_features(gray_grad)
    return glgcm_features

# @jit
def get_glgcm_features(mat):
    '''根据灰度梯度共生矩阵计算纹理特征量，包括小梯度优势，大梯度优势，灰度分布不均匀性，梯度分布不均匀性，能量，灰度平均，梯度平均，
    灰度方差，梯度方差，相关，灰度熵，梯度熵，混合熵，惯性，逆差矩'''
    sum_mat = mat.sum()
    small_grads_dominance = big_grads_dominance = gray_asymmetry = grads_asymmetry = energy = gray_mean = grads_mean = 0
    gray_variance = grads_variance = corelation = gray_entropy = grads_entropy = entropy = inertia = differ_moment = 0
    for i in range(mat.shape[0]):
        gray_variance_temp = 0
        for j in range(mat.shape[1]):
            small_grads_dominance += mat[i][j] / ((j + 1) ** 2)
            big_grads_dominance += mat[i][j] * j ** 2
            energy += mat[i][j] ** 2
            if mat[i].sum() != 0:
                gray_entropy -= mat[i][j] * np.log(mat[i].sum())
            if mat[:, j].sum() != 0:
                grads_entropy -= mat[i][j] * np.log(mat[:, j].sum())
            if mat[i][j] != 0:
                entropy -= mat[i][j] * np.log(mat[i][j])
                inertia += (i - j) ** 2 * np.log(mat[i][j])
            differ_moment += mat[i][j] / (1 + (i - j) ** 2)
            gray_variance_temp += mat[i][j] ** 0.5

        gray_asymmetry += mat[i].sum() ** 2
        gray_mean += i * mat[i].sum() ** 2
        gray_variance += (i - gray_mean) ** 2 * gray_variance_temp
    for j in range(mat.shape[1]):
        grads_variance_temp = 0
        for i in range(mat.shape[0]):
            grads_variance_temp += mat[i][j] ** 0.5
        grads_asymmetry += mat[:, j].sum() ** 2
        grads_mean += j * mat[:, j].sum() ** 2
        grads_variance += (j - grads_mean) ** 2 * grads_variance_temp
    small_grads_dominance /= sum_mat
    big_grads_dominance /= sum_mat
    gray_asymmetry /= sum_mat
    grads_asymmetry /= sum_mat
    gray_variance = gray_variance ** 0.5
    grads_variance = grads_variance ** 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            corelation += (i - gray_mean) * (j - grads_mean) * mat[i][j]
    glgcm_features = [small_grads_dominance, big_grads_dominance, gray_asymmetry, grads_asymmetry, energy, gray_mean, grads_mean,
        gray_variance, grads_variance, corelation, gray_entropy, grads_entropy, entropy, inertia, differ_moment]
    return np.round(glgcm_features, 4)


def get_GGCM_excels(path_cutimg, path_excels_save):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('小梯度优势', cell_overwrite_ok=True)
    sheet2 = f.add_sheet('大梯度优势', cell_overwrite_ok=True)
    sheet3 = f.add_sheet('灰度不均匀性', cell_overwrite_ok=True)
    sheet4 = f.add_sheet('梯度不均匀性', cell_overwrite_ok=True)
    sheet5 = f.add_sheet('能量', cell_overwrite_ok=True)
    sheet6 = f.add_sheet('灰度均值', cell_overwrite_ok=True)
    sheet7 = f.add_sheet('梯度均值', cell_overwrite_ok=True)
    sheet8 = f.add_sheet('灰度均方差', cell_overwrite_ok=True)
    sheet9 = f.add_sheet('梯度均方差', cell_overwrite_ok=True)
    sheet10 = f.add_sheet('相关性', cell_overwrite_ok=True)
    sheet11 = f.add_sheet('灰度熵', cell_overwrite_ok=True)
    sheet12 = f.add_sheet('梯度熵', cell_overwrite_ok=True)
    sheet13 = f.add_sheet('混合熵', cell_overwrite_ok=True)
    sheet14 = f.add_sheet('惯性', cell_overwrite_ok=True)
    sheet15 = f.add_sheet('逆差矩', cell_overwrite_ok=True)

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
        sheet5.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet6.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet7.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet8.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet9.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet10.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet11.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet12.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet13.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet14.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet15.write(0, i, row0[i], set_style('Times New Roman', 220, True))

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

                sheet5.write_merge(q, q, 0, 0, filename1)
                sheet5.write_merge(q, q, 1, 1, filename2)
                sheet5.write_merge(q, q, 2, 2, filename3)

                sheet6.write_merge(q, q, 0, 0, filename1)
                sheet6.write_merge(q, q, 1, 1, filename2)
                sheet6.write_merge(q, q, 2, 2, filename3)

                sheet7.write_merge(q, q, 0, 0, filename1)
                sheet7.write_merge(q, q, 1, 1, filename2)
                sheet7.write_merge(q, q, 2, 2, filename3)

                sheet8.write_merge(q, q, 0, 0, filename1)
                sheet8.write_merge(q, q, 1, 1, filename2)
                sheet8.write_merge(q, q, 2, 2, filename3)

                sheet9.write_merge(q, q, 0, 0, filename1)
                sheet9.write_merge(q, q, 1, 1, filename2)
                sheet9.write_merge(q, q, 2, 2, filename3)

                sheet10.write_merge(q, q, 0, 0, filename1)
                sheet10.write_merge(q, q, 1, 1, filename2)
                sheet10.write_merge(q, q, 2, 2, filename3)

                sheet11.write_merge(q, q, 0, 0, filename1)
                sheet11.write_merge(q, q, 1, 1, filename2)
                sheet11.write_merge(q, q, 2, 2, filename3)

                sheet12.write_merge(q, q, 0, 0, filename1)
                sheet12.write_merge(q, q, 1, 1, filename2)
                sheet12.write_merge(q, q, 2, 2, filename3)

                sheet13.write_merge(q, q, 0, 0, filename1)
                sheet13.write_merge(q, q, 1, 1, filename2)
                sheet13.write_merge(q, q, 2, 2, filename3)

                sheet14.write_merge(q, q, 0, 0, filename1)
                sheet14.write_merge(q, q, 1, 1, filename2)
                sheet14.write_merge(q, q, 2, 2, filename3)

                sheet15.write_merge(q, q, 0, 0, filename1)
                sheet15.write_merge(q, q, 1, 1, filename2)
                sheet15.write_merge(q, q, 2, 2, filename3)

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
                        sheet5.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet6.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet7.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet8.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet9.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet10.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet11.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet12.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet13.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet14.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')
                        sheet15.write_merge(q, q, 2 * i + 3, 2 * i + 3, 'NA')

                        continue
                    # 计算目标图像的灰度共生矩阵值
                    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
                    gray_target = img_as_ubyte(gray_target)
                    glgcm_features_target = glgcm(gray_target, 16, 16)

                    sheet1.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[0])
                    sheet2.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[1])
                    sheet3.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[2])
                    sheet4.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[3])
                    sheet5.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[4])
                    sheet6.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[5])
                    sheet7.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[6])
                    sheet8.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[7])
                    sheet9.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[8])
                    sheet10.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[9])
                    sheet11.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[10])
                    sheet12.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[11])
                    sheet13.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[12])
                    sheet14.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[13])
                    sheet15.write_merge(q, q, 2 * i + 3, 2 * i + 3, glgcm_features_target[14])
                    ##########################

                    [a1, b1, c1] = img_target.shape

                    result_1 = 0
                    result_2 = 0
                    result_3 = 0
                    result_4 = 0
                    result_5 = 0
                    result_6 = 0
                    result_7 = 0
                    result_8 = 0
                    result_9 = 0
                    result_10 = 0
                    result_11 = 0
                    result_12 = 0
                    result_13 = 0
                    result_14 = 0
                    result_15 = 0

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
                        glgcm_features_backg = glgcm(gray_bg, 16, 16)
                        result_1 = result_1 + glgcm_features_backg[0]
                        result_2 = result_2 + glgcm_features_backg[1]
                        result_3 = result_3 + glgcm_features_backg[2]
                        result_4 = result_4 + glgcm_features_backg[3]
                        result_5 = result_5 + glgcm_features_backg[4]
                        result_6 = result_6 + glgcm_features_backg[5]
                        result_7 = result_7 + glgcm_features_backg[6]
                        result_8 = result_8 + glgcm_features_backg[7]
                        result_9 = result_9 + glgcm_features_backg[8]
                        result_10 = result_10 + glgcm_features_backg[9]
                        result_11 = result_11 + glgcm_features_backg[10]
                        result_12 = result_12 + glgcm_features_backg[11]
                        result_13 = result_13 + glgcm_features_backg[12]
                        result_14 = result_14 + glgcm_features_backg[13]
                        result_15 = result_15 + glgcm_features_backg[14]

                        # 以上为修改函数####################################################################
                        cnt = cnt + 1
                        sign = False
                    if sign:
                        everage_1 = 'NA'
                        everage_2 = 'NA'
                        everage_3 = 'NA'
                        everage_4 = 'NA'
                        everage_5 = 'NA'
                        everage_6 = 'NA'
                        everage_7 = 'NA'
                        everage_8 = 'NA'
                        everage_9 = 'NA'
                        everage_10 = 'NA'
                        everage_11 = 'NA'
                        everage_12 = 'NA'
                        everage_13 = 'NA'
                        everage_14 = 'NA'
                        everage_15 = 'NA'
                    else:
                        everage_1 = result_1 / cnt
                        everage_2 = result_2 / cnt
                        everage_3 = result_3 / cnt
                        everage_4 = result_4 / cnt
                        everage_5 = result_5 / cnt
                        everage_6 = result_6 / cnt
                        everage_7 = result_7 / cnt
                        everage_8 = result_8 / cnt
                        everage_9 = result_9 / cnt
                        everage_10 = result_10 / cnt
                        everage_11 = result_11 / cnt
                        everage_12 = result_12 / cnt
                        everage_13 = result_13 / cnt
                        everage_14 = result_14 / cnt
                        everage_15 = result_15 / cnt

                    sheet1.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_1)
                    sheet2.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_2)
                    sheet3.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_3)
                    sheet4.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_4)
                    sheet5.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_5)
                    sheet6.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_6)
                    sheet7.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_7)
                    sheet8.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_8)
                    sheet9.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_9)
                    sheet10.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_10)
                    sheet11.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_11)
                    sheet12.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_12)
                    sheet13.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_13)
                    sheet14.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_14)
                    sheet15.write_merge(q, q, 2 * i + 4, 2 * i + 4, everage_15)
                    # print('end')

    f.save(path_excels_save +'/' + 'excel_GGCM.xls')
    return


# if __name__ =='__main__':
#     fp = 'D:/camouflageProject/matlab2PythonCode/1.png'
#     img = cv2.imread(fp)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     glgcm_features = glgcm(img_gray, 15, 15)
#     print (glgcm_features)

# get_GGCM_excels('D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM',
#                 'D:/Python/Python/WZ_GLDM/webNew2/static/excels_save_3.21')

