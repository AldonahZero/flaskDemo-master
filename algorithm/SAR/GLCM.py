import warnings
import csv
import os
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from .process_pre import nsst_dec
warnings.filterwarnings("ignore")
RESULT_FOLDER = os.path.join('algorithm', 'SAR', 'result')

def contrast_con(P):
    (num_level, num_level2, num_dist, num_angle) = P.shape

    P = P.astype(np.float64)
    glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    weights = (I - J) ** 2;
    weights = weights.reshape((num_level, num_level, 1, 1))
    results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]

    return results


def get_glcm_features(path):
    '''
    :param image: 滤波后图像
    :return: glcm
    energy,entropy,deficit,correlation,contrast——能量、熵、逆差矩、相关度、对比度
    '''
    _,gray = nsst_dec(path)
    image = img_as_ubyte(gray)  # 变成8位无符号整型
    # 这一步类似于数据压缩，因为8位图像含有256个灰度级，这样会导致计算灰度共生矩阵是的计算量过大，因此将其进行压缩成16级，将灰度进行分区
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(image, bins)  # 返回的是一个和image大小一样的矩阵，只是矩阵元素表示的是image中元素在bins中的区间位置，小于0为0,0-16为1，以此类推

    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds,  # 需要进行共生矩阵计算的numpy矩阵
                                      [1],  # 步长
                                      [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],  # 方向角度
                                      levels=max_value,  # 共生矩阵阶数
                                      normed=False, symmetric=False)
    # P[i,j,d,theta]返回的是一个四维矩阵（四个方向），各维代表不同的意义
    energy = greycoprops(matrix_coocurrence, 'ASM')  # 能量
    entropy = greycoprops(matrix_coocurrence, 'entropy')
    deficit = greycoprops(matrix_coocurrence, 'homogeneity')
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    contrast = contrast_con(matrix_coocurrence)
    out = np.zeros((5,4))
    out[0][:] = energy
    out[1][:] = entropy
    out[2][:] = deficit
    out[3][:] = correlation
    out[4][:] = contrast
    # glcm_features_path = RESULT_FOLDER + '/SAR/glcm.csv'
    glcm_features_path = os.path.join(RESULT_FOLDER,'SAR/glcm.csv')
    f = open(glcm_features_path, 'w', newline="",encoding='utf-8-sig')
    csv_writer = csv.writer(f)
    # 构建列表头
    csv_writer.writerow(['扫描角度', '0°', '45°', '90°', '135°'])
    # 写入csv文件内容
    csv_writer.writerow(['energy', energy[0][0], energy[0][1], energy[0][2], energy[0][3]])
    csv_writer.writerow(['entropy', entropy[0][0], entropy[0][1], entropy[0][2], entropy[0][3]])
    csv_writer.writerow(['deficit', deficit[0][0], deficit[0][1], deficit[0][2], deficit[0][3]])
    csv_writer.writerow(['correlation', correlation[0][0], correlation[0][1], correlation[0][2], correlation[0][3]])
    csv_writer.writerow(['contrast', contrast[0][0], contrast[0][1], contrast[0][2], contrast[0][3]])
    f.close()

    return glcm_features_path,out

# if __name__ == '__main__':
#     _,out = get_glcm_features(r'D:\back_dev_flask-master1\static\result\SAR\image_f.png')
#     print(out)
