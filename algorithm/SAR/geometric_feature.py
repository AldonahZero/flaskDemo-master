import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from skimage import measure, morphology
import math
import csv
import os
from .process_pre import otsu_2d
RESULT_FOLDER = os.path.join('algorithm', 'SAR', 'result')

def get_geometric_feature(path):
    _,image = otsu_2d(path)

    bw = morphology.closing(image, morphology.square(10))  # 闭运算

    cleared = bw.copy()  # 复制
    # segmentation.clear_border(cleared)  # 清除与边界相连的目标物

    label_image = measure.label(cleared)  # 连通区域标记

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
    ax0.imshow(cleared, plt.cm.gray)
    ax1.imshow(image)

    for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集

        # 忽略小区域
        if region.area < 100:
            continue

        # 绘制外包矩形
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc - 1, minr - 1), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax1.add_patch(rect)
        area_t = region.area
        c_t = region.perimeter
        E = region.eccentricity
        beita = c_t**2 / (4 * math.pi * area_t)
        res = np.array((area_t,c_t,E,beita))
    # geometric_path = RESULT_FOLDER + '/SAR/geometric.csv'
    geometric_path = os.path.join(RESULT_FOLDER,'SAR','geometric.csv')
    f = open(geometric_path, 'w', newline="",encoding='utf-8-sig')
    csv_writer = csv.writer(f)
    # 构建列表头
    csv_writer.writerow(['面积','周长','离心率','形状复杂度'])
    # 写入csv文件内容
    csv_writer.writerow([area_t,c_t,E,beita])
    f.close()
    return geometric_path, res

# if __name__ == '__main__':
#     path = r'D:\sar\实验图像\report_image\pc\pc2\2021061917(10~43)pc2.tif'
#     get_geometric_feature(path)