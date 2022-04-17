import cv2
import os
import numpy as np
import csv

from .process_pre import nsst_dec
RESULT_FOLDER = os.path.join('algorithm', 'SAR', 'result')


def Simple_DBC(path):
    # img = cv2.imread(path)
    # if img.ndim > 2:
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = nsst_dec(path)
    P = cv2.resize(gray.astype('uint8'), (256, 256))
    G = 256  # 灰度等级
    s = 2 ** np.array([1, 2, 3, 4, 5, 6, 7])  # 格子边
    [M, N] = P.shape
    h = 2 ** np.array([1, 2, 3, 4, 5, 6, 7])  # 网格高度
    Nr = []
    k = 0
    for size in s:
        cell = []
        for i in range(0, M - size + 1, size):
            for j in range(0, N - size + 1, size):
                cell_m = P[i:i + size, j:j + size]
                count = round((cell_m.max() - cell_m.min()) / h[k]) + 1
                cell.append(count)
        k += 1
        Nr.append(sum(cell))
    y = np.log(Nr)
    x = np.log(G / s)
    coeffs = np.polyfit(x, y, 1)

    # fractal_features_path = RESULT_FOLDER + '/SAR/fractal.csv'
    fractal_features_path = os.path.join(RESULT_FOLDER, 'SAR','fractal.csv')
    f = open(fractal_features_path, 'w', encoding='utf-8-sig', newline="")
    csv_writer = csv.writer(f)
    # 构建列表头
    csv_writer.writerow(['名称', 'Fractal'])
    # 写入csv文件内容
    csv_writer.writerow(['值', abs(coeffs[0])])
    f.close()

    return fractal_features_path, abs(coeffs[0])

# if __name__ == '__main__':
#     print(Simple_DBC(r'D:\back_dev_flask-master\static\result\SAR\image_f.png'))
