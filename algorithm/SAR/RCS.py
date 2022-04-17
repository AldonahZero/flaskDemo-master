import cv2
import math
import os
import matplotlib.patches as mpatches
import numpy as np
import csv
import warnings
from skimage import measure
from skimage.transform import radon
from .process_pre import otsu_2d

RESULT_FOLDER = os.path.join('algorithm', 'SAR', 'result')

warnings.filterwarnings("ignore")


def RCS(path1):
    '''
    :param I: 输入图像
    :param I_b: 输入图像二值图
    :return: RCS特征曲线图存储路径
    '''
    # path2 = RESULT_FOLDER + '/SAR/image_b.png'
    I = cv2.imread(path1)
    # I_b = cv2.imread(path2)
    if I.ndim > 2:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # if I_b.ndim > 2:
    #     I_b = cv2.cvtColor(I_b, cv2.COLOR_BGR2GRAY)
    _, I_b = otsu_2d(path1)
    h, w = I.shape
    # 如果图像分辨率太高，改变图像尺寸减小运算量
    if (w > 1000):
        I = cv2.resize(I, (1000, int((h / w) * 1000)))
    I = I - np.mean(I)  # Demean; make the brightness extend above and below zero

    # 进行radon变换
    sinogram = radon(I)
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)

    ### 将图片进行旋转
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 90 - rotation, 1)
    img_o = cv2.warpAffine(I, M, (w, h))
    # img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
    img = cv2.warpAffine(I_b, M, (w, h))
    # plt.imshow(img_o,plt.cm.gray)
    # plt.show()
    # plt.imshow(img, plt.cm.gray)
    # plt.show()

    cleared = img.copy()  # 复制

    label_image = measure.label(cleared)  # 连通区域标记

    ###目标外接矩
    area = []
    for region in measure.regionprops(label_image):
        area_tem = region.area
        area.append(area_tem)
    for region in measure.regionprops(label_image):
        # 寻找最大连通区域画外接矩
        if region.area == max(area):
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc - 1, minr - 1), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
    #         ax1.add_patch(rect)
    # fig.tight_layout()
    # plt.show()

    ###根据外接矩计算RCS
    max_rec1 = minc
    max_rec2 = minr
    max_rec3 = maxc - minc
    max_rec4 = maxr - minr

    w1 = 21
    w2 = min(max_rec3, max_rec4)
    if w2 % 2 == 0:
        w2 += 1

    t = 1
    RCS = []

    i = math.ceil(max_rec2 + (w2 - 1) / 2)
    a = math.ceil(max_rec1 + (w1 - 1) / 2)
    b = math.ceil(max_rec1 + max_rec3 - (w1 - 1) / 2)
    for j in range(a, b + 1):
        W = img_o[int((i - (w2 - 1) / 2)):int((i + 1 + (w2 - 1) / 2)),
            int((j - (w1 - 1) / 2)):int((j + 1 + (w1 - 1) / 2))]
        # print(i - (w2 - 1) / 2, i + (w2 - 1) / 2,j - (w1 - 1) / 2, j + (w1 - 1) / 2)
        RCS.append(W.sum() / (w1 * w2))
    length = len(RCS)
    axis = []
    for i in range(length):
        axis.append(i)

    # RCS_feature_path = RESULT_FOLDER + '/SAR/RCS.csv'
    RCS_feature_path = os.path.join(RESULT_FOLDER, 'SAR', 'RCS.csv')
    f = open(RCS_feature_path, 'w', newline="")
    csv_writer = csv.writer(f)
    # # 构建列表头
    csv_writer.writerow(axis)
    # 写入csv文件内容
    csv_writer.writerow(RCS)
    f.close()

    return RCS_feature_path, RCS

# if __name__ == '__main__':
#     print(RCS(r'D:\back_dev_flask-master\static\uploads\SAR\HB19975.JPG'))
