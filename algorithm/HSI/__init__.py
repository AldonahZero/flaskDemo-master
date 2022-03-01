import numpy as np
import cv2


#  这个数据大小 是写死的  以后如果可以的话  要改一下...
def load_data(image_path):
    raw_image = np.fromfile(image_path, "float32")
    image = np.zeros((1057, 960, 176))
    for row in range(0, 1057):
        for dim in range(0, 176):
            image[row, :, dim] = raw_image[(dim + row * 176) * 960:(dim + 1 + row * 176) * 960]
    return image


# 转换成二维的时候   是一行一行来的
def to_2d_array(array):
    [m, n, p] = array.shape
    res = np.zeros((p, m*n))
    for i in range(p):
        res[i, :] = np.reshape(array[:, :, i], (1, m*n))
    return res


# 显示伪彩图像
def show_image(image_path):
    red_band = 76
    blue_band = 15
    green_band = 46
    print(1)
    image = load_data(image_path)
    [m, n, p] = image.shape
    print(m, n, p)
    img_r = image[:, :, red_band]
    img_g = image[:, :, green_band]

    img_b = image[:, :, blue_band]
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    [m, n, p] = pseudo_image.shape
    print(m, n, p)
    pseudo_image = pseudo_image * 255.0
    #  输出路径
    out_path = "E:/HSI/test.png"
    cv2.imwrite(out_path, pseudo_image)
    '''out_path = "E:/dataTest/test.png"
    cv2.imwrite(out_path, pseudo_image)
    cv2.imshow("test", pseudo_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return out_path
