
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
    '''plt.figure()
    plt.axis("off")
    plt.imshow(bar)

    io.imsave("F:\\data_test\\2.jpg", bar)
    return a_per, b_color'''
    return bar, per, color


# path_cutimg为cutimg分割结果 path_excels_save为表格存储路径
def get_main_color_excels(path_cutimg, path_excels_save):
    f = xlwt.Workbook()

    sheet1 = f.add_sheet('main_color', cell_overwrite_ok=True)
    sheet2 = f.add_sheet('percent', cell_overwrite_ok=True)

    row0 = ["c/nc", "v/i", "h",
            "angel0_target", "angel0_back", "angel1_target", "angel1_back", "angel2_target", "angel2_back",
            "angel3_target", "angel3_back", "angel4_target", "angel4_back"
                                                             "angel5_target", "angel5_back", "angel6_target",
            "angel6_back", "angel7_target", "angel7_back",
            "angel8_target", "angel8_back", "angel9_target", "angel9_back"
            ]
    row1 = ["c/nc", "v/i", "h",
            "angel0_target", "angel0_back", "angel1_target", "angel1_back", "angel2_target", "angel2_back",
            "angel3_target", "angel3_back", "angel4_target", "angel4_back",
            "angel5_target", "angel5_back", "angel6_target", "angel6_back", "angel7_target", "angel7_back",
            "angel8_target", "angel8_back", "angel9_target", "angel9_back"
            ]
    for i in range(0, 3):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        sheet2.write(0, i, row1[i], set_style('Times New Roman', 220, True))

    # print(len(row0))

    for i in range(3, len(row0)):
        sheet1.write(0, i * 4 - 9, row0[i], set_style('Times New Roman', 220, True))
        sheet2.write(0, i * 2 - 3, row1[i], set_style('Times New Roman', 220, True))

    q = 1
    # 改一下这下面这两个 参数
    # path = 'E:/data/chuzhou_data'
    # 待处理图片路径
    # path = 'D:\\data\\chuzhou_data'
    path = path_cutimg;
    # 主色提取图片保存路径
    # out_path1 = 'D:\\data\\chuzhou_data_1'
    out_path1 = path_excels_save
    cnt = 0
    for filename1 in os.listdir(path):
        path1 = path + '/' + filename1
        for filename2 in os.listdir(path1):
            path2 = path1 + '/' + filename2
            for filename3 in os.listdir(path2):
                path3 = path2 + '/' + filename3

                sheet1.write_merge(q, q, 0, 0, filename1)
                sheet1.write_merge(q, q, 1, 1, filename2)
                sheet1.write_merge(q, q, 2, 2, filename3)
                sheet2.write_merge(q, q, 0, 0, filename1)
                sheet2.write_merge(q, q, 1, 1, filename2)
                sheet2.write_merge(q, q, 2, 2, filename3)
                for i in range(10):
                    out_path = out_path1 + '/' + filename1 + '/' + filename2 + '/' + filename3 + '/' + str(i)
                    target_out_path = out_path + 'target.jpg'
                    back_out_path = out_path + 'back.jpg'
                    target_name = str(i) + str(4) + '.jpg'
                    img_target = cv2.imread(path3 + '/' + target_name)
                    if img_target is None:
                        sheet1.write_merge(q, q, 8 * i + 3, 8 * i + 3, 'NA')
                        continue
                    # 计算目标
                    img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
                    img_target = img_target.reshape((img_target.shape[0] * img_target.shape[1], 3))
                    res, per, color_main = c_main(img_target)
                    # plt.figure()
                    # plt.axis("off")
                    # plt.imshow(res)
                    # io.imsave(target_out_path, res)
                    # plt.close('all')

                    img_back = np.zeros((0, 3))
                    for j in range(9):
                        if j == 4:
                            continue
                        backg_name = str(i) + str(j) + '.jpg'
                        if not os.path.exists(path3 + '/' + backg_name):
                            continue
                        img_backg = cv2.imread(path3 + '/' + backg_name)
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
                        # plt.figure()
                        # plt.axis("off")
                        # plt.imshow(res)
                        # io.imsave(back_out_path, res_back)
                        # plt.close('all')
                    # print(i)
                    sheet1.write_merge(q, q, 8 * i + 3, 8 * i + 3, color_main[0][0])
                    sheet1.write_merge(q, q, 8 * i + 4, 8 * i + 4, color_main[0][1])
                    sheet1.write_merge(q, q, 8 * i + 5, 8 * i + 5, color_main[0][2])

                    sheet1.write_merge(q + 1, q + 1, 8 * i + 3, 8 * i + 3, color_main[1][0])
                    sheet1.write_merge(q + 1, q + 1, 8 * i + 4, 8 * i + 4, color_main[1][1])
                    sheet1.write_merge(q + 1, q + 1, 8 * i + 5, 8 * i + 5, color_main[1][2])

                    sheet1.write_merge(q + 2, q + 2, 8 * i + 3, 8 * i + 3, color_main[2][0])
                    sheet1.write_merge(q + 2, q + 2, 8 * i + 4, 8 * i + 4, color_main[2][1])
                    sheet1.write_merge(q + 2, q + 2, 8 * i + 5, 8 * i + 5, color_main[2][2])

                    sheet1.write_merge(q + 3, q + 3, 8 * i + 3, 8 * i + 3, color_main[3][0])
                    sheet1.write_merge(q + 3, q + 3, 8 * i + 4, 8 * i + 4, color_main[3][1])
                    sheet1.write_merge(q + 3, q + 3, 8 * i + 5, 8 * i + 5, color_main[3][2])

                    sheet1.write_merge(q + 4, q + 4, 8 * i + 3, 8 * i + 3, color_main[4][0])
                    sheet1.write_merge(q + 4, q + 4, 8 * i + 4, 8 * i + 4, color_main[4][1])
                    sheet1.write_merge(q + 4, q + 4, 8 * i + 5, 8 * i + 5, color_main[4][2])

                    sheet1.write_merge(q, q, 8 * i + 7, 8 * i + 7, color_main_back[0][0])
                    sheet1.write_merge(q, q, 8 * i + 8, 8 * i + 8, color_main_back[0][1])
                    sheet1.write_merge(q, q, 8 * i + 9, 8 * i + 9, color_main_back[0][2])

                    sheet1.write_merge(q + 1, q + 1, 8 * i + 7, 8 * i + 7, color_main_back[1][0])
                    sheet1.write_merge(q + 1, q + 1, 8 * i + 8, 8 * i + 8, color_main_back[1][1])
                    sheet1.write_merge(q + 1, q + 1, 8 * i + 9, 8 * i + 9, color_main_back[1][2])

                    sheet1.write_merge(q + 2, q + 2, 8 * i + 7, 8 * i + 7, color_main_back[2][0])
                    sheet1.write_merge(q + 2, q + 2, 8 * i + 8, 8 * i + 8, color_main_back[2][1])
                    sheet1.write_merge(q + 2, q + 2, 8 * i + 9, 8 * i + 9, color_main_back[2][1])

                    sheet1.write_merge(q + 3, q + 3, 8 * i + 7, 8 * i + 7, color_main_back[3][0])
                    sheet1.write_merge(q + 3, q + 3, 8 * i + 8, 8 * i + 8, color_main_back[3][1])
                    sheet1.write_merge(q + 3, q + 3, 8 * i + 9, 8 * i + 9, color_main_back[3][2])

                    sheet1.write_merge(q + 4, q + 4, 8 * i + 7, 8 * i + 7, color_main_back[4][0])
                    sheet1.write_merge(q + 4, q + 4, 8 * i + 8, 8 * i + 8, color_main_back[4][1])
                    sheet1.write_merge(q + 4, q + 4, 8 * i + 9, 8 * i + 9, color_main_back[4][2])

                    sheet2.write_merge(q, q, 4 * i + 3, 4 * i + 3, per[0])
                    sheet2.write_merge(q + 1, q + 1, 4 * i + 3, 4 * i + 3, per[1])
                    sheet2.write_merge(q + 2, q + 2, 4 * i + 3, 4 * i + 3, per[2])
                    sheet2.write_merge(q + 3, q + 3, 4 * i + 3, 4 * i + 3, per[3])
                    sheet2.write_merge(q + 4, q + 4, 4 * i + 3, 4 * i + 3, per[4])

                    sheet2.write_merge(q, q, 4 * i + 5, 4 * i + 5, per_back[0])
                    sheet2.write_merge(q + 1, q + 1, 4 * i + 5, 4 * i + 5, per_back[1])
                    sheet2.write_merge(q + 2, q + 2, 4 * i + 5, 4 * i + 5, per_back[2])
                    sheet2.write_merge(q + 3, q + 3, 4 * i + 5, 4 * i + 5, per_back[3])
                    sheet2.write_merge(q + 4, q + 4, 4 * i + 5, 4 * i + 5, per_back[4])
                q = q + 6

    # 主色提取颜色RGB值以及占比表格保存路径
    f.save(path_excels_save + '/' + 'excel_main_color.xls')
    return

'''
img_target = cv2.imread(r'D:\Python\Python\WZ_GLDM\web\static\images_GLCM_original\images_camouflage\mix\20m\2.JPG')
img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
img_target = img_target.reshape((img_target.shape[0] * img_target.shape[1], 3))
res, per, color_main = c_main(img_target)
plt.figure()
plt.axis("off")
plt.imshow(res)
plt.show()
'''

# 批处理
'''
img_target = img_target.reshape((img_target.shape[0] * img_target.shape[1], 3))
res, per, color_main = c_main(img_target)
plt.figure()
plt.axis("off")

f = xlwt.Workbook()

sheet1 = f.add_sheet('main_color', cell_overwrite_ok=True)
sheet2 = f.add_sheet('percent', cell_overwrite_ok=True)


row0 = ["c/nc", "v/i", "h",
        "angel0_target", "angel0_back", "angel1_target", "angel1_back", "angel2_target", "angel2_back",
        "angel3_target", "angel3_back", "angel4_target", "angel4_back"
        "angel5_target", "angel5_back", "angel6_target", "angel6_back", "angel7_target", "angel7_back",
        "angel8_target", "angel8_back", "angel9_target", "angel9_back"
        ]
row1 = ["c/nc", "v/i", "h",
        "angel0_target", "angel0_back", "angel1_target", "angel1_back", "angel2_target", "angel2_back",
        "angel3_target", "angel3_back", "angel4_target", "angel4_back",
        "angel5_target", "angel5_back", "angel6_target", "angel6_back", "angel7_target", "angel7_back",
        "angel8_target", "angel8_back", "angel9_target", "angel9_back"
        ]
for i in range(0, 3):
    sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
    sheet2.write(0, i, row1[i], set_style('Times New Roman', 220, True))

for i in range(3, len(row0)):
    sheet1.write(0, i * 4 - 9, row0[i], set_style('Times New Roman', 220, True))
    sheet2.write(0, i * 2 - 3, row1[i], set_style('Times New Roman', 220, True))


q = 2
# 改一下这下面这两个 参数
# path = 'E:/data/chuzhou_data'
# 待处理图片路径
path = 'D:\\data\\chuzhou_data'
# 主色提取图片保存路径
out_path1 = 'D:\\data\\chuzhou_data_1'
cnt = 0
for filename1 in os.listdir(path):
    path1 = path + '/' + filename1
    for filename2 in os.listdir(path1):
        path2 = path1 + '/' + filename2
        for filename3 in os.listdir(path2):
            path3 = path2 + '/' + filename3

            sheet1.write_merge(q, q, 0, 0, filename1)
            sheet1.write_merge(q, q, 1, 1, filename2)
            sheet1.write_merge(q, q, 2, 2, filename3)
            sheet2.write_merge(q, q, 0, 0, filename1)
            sheet2.write_merge(q, q, 1, 1, filename2)
            sheet2.write_merge(q, q, 2, 2, filename3)
            for i in range(10):
                out_path = out_path1 + '/' + filename1 + '/' + filename2 + '/' + filename3 + '/' + str(i)
                target_out_path = out_path + 'target.jpg'
                back_out_path = out_path + 'back.jpg'
                target_name = str(i) + str(4) + '.jpg'
                img_target = cv2.imread(path3 + '/' + target_name)
                if img_target is None:
                    sheet1.write_merge(q, q, 8 * i + 3, 8 * i + 3, 'NA')
                    continue
                # 计算目标
                img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
                img_target = img_target.reshape((img_target.shape[0] * img_target.shape[1], 3))
                res, per, color_main = c_main(img_target)
                plt.figure()
                plt.axis("off")
                plt.imshow(res)
                io.imsave(target_out_path, res)
                plt.close('all')

                img_back = np.zeros((0, 3))
                for j in range(9):
                    if j == 4:
                        continue
                    backg_name = str(i) + str(j) + '.jpg'
                    if not os.path.exists(path3 + '/' + backg_name):
                        continue
                    img_backg = cv2.imread(path3 + '/' + backg_name)
                    if img_backg is None:
                        continue
                    img_backg = cv2.cvtColor(img_backg, cv2.COLOR_BGR2RGB)
                    img_backg = img_backg.reshape((img_backg.shape[0] * img_backg.shape[1], 3))
                    len = img_backg.shape[0] + img_back.shape[0]
                    temp = np.zeros((len, 3))
                    temp[0: img_back.shape[0], :] = img_back
                    temp[img_back.shape[0]:len, :] = img_backg
                    img_back = temp
                if img_back.shape[0] == 0:
                    per_back = [-1, -1, -1, -1, -1]
                    color_main_back = np.zeros((5, 3))
                else:
                    res_back, per_back, color_main_back = c_main(img_back)
                    plt.figure()
                    plt.axis("off")
                    plt.imshow(res)
                    io.imsave(back_out_path, res_back)
                    plt.close('all')
                print(i)
                sheet1.write_merge(q, q, 8 * i + 3, 8 * i + 3, color_main[0][0])
                sheet1.write_merge(q, q, 8 * i + 4, 8 * i + 4, color_main[0][1])
                sheet1.write_merge(q, q, 8 * i + 5, 8 * i + 5, color_main[0][2])

                sheet1.write_merge(q + 1, q + 1, 8 * i + 3, 8 * i + 3, color_main[1][0])
                sheet1.write_merge(q + 1, q + 1, 8 * i + 4, 8 * i + 4, color_main[1][1])
                sheet1.write_merge(q + 1, q + 1, 8 * i + 5, 8 * i + 5, color_main[1][2])

                sheet1.write_merge(q + 2, q + 2, 8 * i + 3, 8 * i + 3, color_main[2][0])
                sheet1.write_merge(q + 2, q + 2, 8 * i + 4, 8 * i + 4, color_main[2][1])
                sheet1.write_merge(q + 2, q + 2, 8 * i + 5, 8 * i + 5, color_main[2][2])

                sheet1.write_merge(q + 3, q + 3, 8 * i + 3, 8 * i + 3, color_main[3][0])
                sheet1.write_merge(q + 3, q + 3, 8 * i + 4, 8 * i + 4, color_main[3][1])
                sheet1.write_merge(q + 3, q + 3, 8 * i + 5, 8 * i + 5, color_main[3][2])

                sheet1.write_merge(q + 4, q + 4, 8 * i + 3, 8 * i + 3, color_main[4][0])
                sheet1.write_merge(q + 4, q + 4, 8 * i + 4, 8 * i + 4, color_main[4][1])
                sheet1.write_merge(q + 4, q + 4, 8 * i + 5, 8 * i + 5, color_main[4][2])

                sheet1.write_merge(q, q, 8 * i + 7, 8 * i + 7, color_main_back[0][0])
                sheet1.write_merge(q, q, 8 * i + 8, 8 * i + 8, color_main_back[0][1])
                sheet1.write_merge(q, q, 8 * i + 9, 8 * i + 9, color_main_back[0][2])

                sheet1.write_merge(q + 1, q + 1, 8 * i + 7, 8 * i + 7, color_main_back[1][0])
                sheet1.write_merge(q + 1, q + 1, 8 * i + 8, 8 * i + 8, color_main_back[1][1])
                sheet1.write_merge(q + 1, q + 1, 8 * i + 9, 8 * i + 9, color_main_back[1][2])

                sheet1.write_merge(q + 2, q + 2, 8 * i + 7, 8 * i + 7, color_main_back[2][0])
                sheet1.write_merge(q + 2, q + 2, 8 * i + 8, 8 * i + 8, color_main_back[2][1])
                sheet1.write_merge(q + 2, q + 2, 8 * i + 9, 8 * i + 9, color_main_back[2][1])

                sheet1.write_merge(q + 3, q + 3, 8 * i + 7, 8 * i + 7, color_main_back[3][0])
                sheet1.write_merge(q + 3, q + 3, 8 * i + 8, 8 * i + 8, color_main_back[3][1])
                sheet1.write_merge(q + 3, q + 3, 8 * i + 9, 8 * i + 9, color_main_back[3][2])

                sheet1.write_merge(q + 4, q + 4, 8 * i + 7, 8 * i + 7, color_main_back[4][0])
                sheet1.write_merge(q + 4, q + 4, 8 * i + 8, 8 * i + 8, color_main_back[4][1])
                sheet1.write_merge(q + 4, q + 4, 8 * i + 9, 8 * i + 9, color_main_back[4][2])

                sheet2.write_merge(q, q, 4 * i + 3, 4 * i + 3, per[0])
                sheet2.write_merge(q + 1, q + 1, 4 * i + 3, 4 * i + 3, per[1])
                sheet2.write_merge(q + 2, q + 2, 4 * i + 3, 4 * i + 3, per[2])
                sheet2.write_merge(q + 3, q + 3, 4 * i + 3, 4 * i + 3, per[3])
                sheet2.write_merge(q + 4, q + 4, 4 * i + 3, 4 * i + 3, per[4])

                sheet2.write_merge(q, q, 4 * i + 5, 4 * i + 5, per_back[0])
                sheet2.write_merge(q + 1, q + 1, 4 * i + 5, 4 * i + 5, per_back[1])
                sheet2.write_merge(q + 2, q + 2, 4 * i + 5, 4 * i + 5, per_back[2])
                sheet2.write_merge(q + 3, q + 3, 4 * i + 5, 4 * i + 5, per_back[3])
                sheet2.write_merge(q + 4, q + 4, 4 * i + 5, 4 * i + 5, per_back[4])
            q = q + 6

# 主色提取颜色RGB值以及占比表格保存路径
f.save('D:\\data\\chuzhou_data\\main_color.xls')


# f.save('E:/data/point.xls')
'''

# get_main_color_excels('D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM',
#                       'D:/Python/Python/WZ_GLDM/webNew2/static/excels_save_3.21')