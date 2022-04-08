
import numpy as np
import os
import xlwt
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils
import cv2

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


def centroid_histogram(clt):

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    percent_1 = []
    color_1 = []
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
        percent_1.append(percent)
        color_1.append(color)
    # return the bar chart
    return bar, percent_1, color_1


def c_main(img):

    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img
    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
    plt.close("all")

    # reshape the image to be a list of pixels
    # image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters=5)
    clt.fit(image)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)
    bar, per, color = plot_colors(hist, clt.cluster_centers_)
    # show our color bart
    return bar, per, color


# path_cutimg为cutimg分割结果 path_excels_save为表格存储路径
def myMainColor_excelSave(path_cutimg, excels_color_main_color):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('主色提取', cell_overwrite_ok=True)

    row0 = ["主色占比", "目标区域R值", "目标区域G值", "目标区域B值", "目标区域颜色占比", "背景区域R值",
            "背景区域G值", "背景区域B值", "背景区域颜色占比"]

    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    # path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'
    # excels_color_main_color = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/color_main_color/'

    cnt = 0

    img_target = cv2.imread(path_cutimg + '14.jpg')
    img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
    img_target = img_target.reshape((img_target.shape[0] * img_target.shape[1], 3))
    res, per, color_main = c_main(img_target)

    img_back = np.zeros((0, 3))
    i = 1
    for j in range(9):
        if j == 4:
            continue
        backg_name = str(i) + str(j) + '.jpg'
        if not os.path.exists(path_cutimg + '/' + backg_name):
            continue
        img_backg = cv2.imread(path_cutimg + '/' + backg_name)
        if img_backg is None:
            continue
        img_backg = cv2.cvtColor(img_backg, cv2.COLOR_BGR2RGB)
        img_backg = img_backg.reshape((img_backg.shape[0] * img_backg.shape[1], 3))
        lens = img_backg.shape[0] + img_back.shape[0]
        temp = np.zeros((lens, 3))
        temp[0: img_back.shape[0], :] = img_back
        temp[img_back.shape[0]:lens, :] = img_backg
        img_back = temp
    if img_back.shape[0] == 0:
        per_back = [-1, -1, -1, -1, -1]
        color_main_back = np.zeros((5, 3))
    else:
        res_back, per_back, color_main_back = c_main(img_back)

    sheet1.write_merge(1, 1, 1, 1, color_main[0][0])
    sheet1.write_merge(1, 1, 2, 2, color_main[0][1])
    sheet1.write_merge(1, 1, 3, 3, color_main[0][2])

    sheet1.write_merge(2, 2, 1, 1, color_main[1][0])
    sheet1.write_merge(2, 2, 2, 2, color_main[1][1])
    sheet1.write_merge(2, 2, 3, 3, color_main[1][2])

    sheet1.write_merge(3, 3, 1, 1, color_main[2][0])
    sheet1.write_merge(3, 3, 2, 2, color_main[2][1])
    sheet1.write_merge(3, 3, 3, 3, color_main[2][2])

    sheet1.write_merge(4, 4, 1, 1, color_main[3][0])
    sheet1.write_merge(4, 4, 2, 2, color_main[3][1])
    sheet1.write_merge(4, 4, 3, 3, color_main[3][2])

    sheet1.write_merge(5, 5, 1, 1, color_main[4][0])
    sheet1.write_merge(5, 5, 2, 2, color_main[4][1])
    sheet1.write_merge(5, 5, 3, 3, color_main[4][2])


    sum_per = np.sum(per)
    sheet1.write_merge(1, 1, 4, 4, per[0] / sum_per)
    sheet1.write_merge(2, 2, 4, 4, per[1] / sum_per)
    sheet1.write_merge(3, 3, 4, 4, per[2] / sum_per)
    sheet1.write_merge(4, 4, 4, 4, per[3] / sum_per)
    sheet1.write_merge(5, 5, 4, 4, per[4] / sum_per)


    sheet1.write_merge(1, 1, 5, 5, color_main_back[0][0])
    sheet1.write_merge(1, 1, 6, 6, color_main_back[0][1])
    sheet1.write_merge(1, 1, 7, 7, color_main_back[0][2])

    sheet1.write_merge(2, 2, 5, 5, color_main_back[1][0])
    sheet1.write_merge(2, 2, 6, 6, color_main_back[1][1])
    sheet1.write_merge(2, 2, 7, 7, color_main_back[1][2])

    sheet1.write_merge(3, 3, 5, 5, color_main_back[2][0])
    sheet1.write_merge(3, 3, 6, 6, color_main_back[2][1])
    sheet1.write_merge(3, 3, 7, 7, color_main_back[2][2])

    sheet1.write_merge(4, 4, 5, 5, color_main_back[3][0])
    sheet1.write_merge(4, 4, 6, 6, color_main_back[3][1])
    sheet1.write_merge(4, 4, 7, 7, color_main_back[3][2])

    sheet1.write_merge(5, 5, 5, 5, color_main_back[4][0])
    sheet1.write_merge(5, 5, 6, 6, color_main_back[4][1])
    sheet1.write_merge(5, 5, 7, 7, color_main_back[4][2])


    sum_per_back = np.sum(per_back)
    sheet1.write_merge(1, 1, 8, 8, per_back[0] / sum_per_back)
    sheet1.write_merge(2, 2, 8, 8, per_back[1] / sum_per_back)
    sheet1.write_merge(3, 3, 8, 8, per_back[2] / sum_per_back)
    sheet1.write_merge(4, 4, 8, 8, per_back[3] / sum_per_back)
    sheet1.write_merge(5, 5, 8, 8, per_back[4] / sum_per_back)

    # print(color_main, color_main_back, per, per_back)


    # 主色提取颜色RGB值以及占比表格保存路径
    f.save(excels_color_main_color + 'excel_main_color.xls')
    return excels_color_main_color

