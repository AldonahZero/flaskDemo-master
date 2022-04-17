# 图像拼接
import cv2
import os
import numpy as np
RESULT_FOLDER = os.path.join('algorithm', 'SAR', 'result')


def warp_corner(H, src):
    '''
    :param H: 单应矩阵
    :param src: 透视变化的图像
    :return: 透视变化后的四个角，左上角开始，逆时钟
    '''

    warp_points = []
    # 图像左上角，左下角
    src_left_up = np.array([0, 0, 1])
    src_left_down = np.array([0, src.shape[0], 1])

    # 图像右上角，右下角
    src_right_up = np.array([src.shape[1], 0, 1])
    src_right_down = np.array([src.shape[1], src.shape[0], 1])

    # 透视变化后的左上角，左下角
    warp_left_up = H.dot(src_left_up)
    left_up = warp_left_up[0:2] / warp_left_up[2]
    warp_points.append(left_up)
    warp_left_down = H.dot(src_left_down)
    left_down = warp_left_down[0:2] / warp_left_down[2]
    warp_points.append(left_down)

    # 透视变化后的右上角，右下角
    warp_right_up = H.dot(src_right_up)
    right_up = warp_right_up[0:2] / warp_right_up[2]
    warp_points.append(right_up)
    warp_right_down = H.dot(src_right_down)
    right_down = warp_right_down[0:2] / warp_right_down[2]
    warp_points.append(right_down)
    return warp_points


def optim_mask(mask, warp_point):
    min_left_x = min(warp_point[0][0], warp_point[1][0])
    left_margin = mask.shape[1] - min_left_x
    points_zeros = np.where(mask == 0)
    x_indexs = points_zeros[1]
    alpha = (left_margin - (x_indexs - min_left_x)) / left_margin
    mask[points_zeros] = alpha
    return mask


def Seam_Left_Right(left, imagewarp, H, warp_point, with_optim_mask=False):
    '''
    :param left: 拼接的左图像
    :param imagewarp: 透视变化后的右图像
    :param H: 单应矩阵
    :param warp_point: 透视变化后的四个顶点
    :param with_optim_mask: 是否需要对拼接后的图像进行优化
    :return:
    '''
    w = left.shape[1]
    mask = imagewarp[:, 0:w]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask != 0] = 1
    mask[mask == 0] = 0
    mask = 1 - mask
    mask = np.float32(mask)

    if with_optim_mask == True:
        mask = optim_mask(mask, warp_point)
    mask_rgb = np.stack([mask, mask, mask], axis=2)
    tt = np.uint8((1 - mask_rgb) * 255)
    left = left * mask_rgb + imagewarp[:, 0:w] * (1 - mask_rgb)
    imagewarp[:, 0:w] = left
    return np.uint8(imagewarp)


def image_stitching(path1, path2, RIGHT_LEFT='1'):
    '''
    :param img1: 参考图像
    :param img2: 待配准图像
    :param RIGHT_LEFT: 判断两幅图像是左右拼接还是上下拼接，若为上下拼接，需对参考图像进行旋转
    :return: 拼接后图像存储路径
    '''
    left = cv2.imread(path1)
    right = cv2.imread(path2)
    if RIGHT_LEFT == '2':
        left = cv2.rotate(left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # 提取左右图像的surf特征点
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
    left_kps, left_dess = detector.detectAndCompute(left_gray, None)
    right_kps, right_dess = detector.detectAndCompute(right_gray, None)

    # 利用knn对左右图像的特征点进行匹配
    matcher = cv2.FlannBasedMatcher_create()
    knn_matchers = matcher.knnMatch(left_dess, right_dess, 2)
    good_keypoints = []

    # 挑出好的匹配点
    for m, n in knn_matchers:
        if m.distance < 0.4 * n.distance:
            good_keypoints.append(m)
    left_points = np.zeros(shape=(len(good_keypoints), 2), dtype=np.float32)
    right_points = np.zeros(shape=(len(good_keypoints), 2), dtype=np.float32)
    outimg = np.zeros(shape=(right.shape[0], right.shape[0] + left.shape[0], 3), dtype=np.uint8)
    cv2.drawMatches(left, left_kps, right, right_kps, good_keypoints, outimg)
    # cv2.imshow('hks',outimg)
    # cv2.waitKey(0)
    for i in range(len(good_keypoints)):
        left_points[i][0] = left_kps[good_keypoints[i].queryIdx].pt[0]
        left_points[i][1] = left_kps[good_keypoints[i].queryIdx].pt[1]
        right_points[i][0] = right_kps[good_keypoints[i].trainIdx].pt[0]
        right_points[i][1] = right_kps[good_keypoints[i].trainIdx].pt[1]

    # 求取单应矩阵
    H, _ = cv2.findHomography(right_points, left_points)

    # 求出右图像的透视变化顶点
    warp_point = warp_corner(H, right)
    # 求出右图像的透视变化图像
    imagewarp = cv2.warpPerspective(right, H, (left.shape[1] + right.shape[1], left.shape[0]))
    # 对左右图像进行拼接，返回最后的拼接图像
    image_seam_optim = Seam_Left_Right(left, imagewarp, H, warp_point, with_optim_mask=True)
    if RIGHT_LEFT == '2':
        image_seam_optim = cv2.rotate(image_seam_optim, cv2.ROTATE_90_CLOCKWISE)
    # cv2.namedWindow('image_seam_optim', cv2.WINDOW_NORMAL)
    # cv2.imshow('image_seam_optim', image_seam_optim)
    # stitching_path = RESULT_FOLDER + '/SAR/image_stitching.png'
    stitching_path = os.path.join(RESULT_FOLDER,'SAR/image_stitching.png')
    cv2.imwrite(stitching_path, image_seam_optim)
    # cv2.waitKey(0)
    # cv2.imwrite('../sar_after_pj2.jpg', image_seam_optim)
    return stitching_path

# if __name__ == '__main__':
#     path1 = r'D:\back_dev_flask-master\static\uploads\SAR\49_reverse.png'
#     path2 = r'D:\back_dev_flask-master\static\uploads\SAR\48.png'
#     image_stitching(path1,path2,'2')
