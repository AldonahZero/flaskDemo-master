import cv2

def kaze_distance(img1, img2): # 计算两幅图kaze特征距离
    kaze = cv2.KAZE_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = kaze.detect(img1, None)
    kp2 = kaze.detect(img2, None)

    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # ## Create flann matcher
    # FLANN_INDEX_KDTREE = 1 # bug: flann enums are missing
    # flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # #matcher = cv2.FlannBasedMatcher_create()
    # matcher = cv2.FlannBasedMatcher(flann_params, {})
    # Brute Force matcher with Hamming distance

    ## Ratio test
    # print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    Max = 0
    Min = 999
    count = 0
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.85 * m2.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            # print(i, pt1, pt2)
            count = count + 1
            a = ((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[0] - pt2[0]) * (pt1[0] - pt2[0])) ** 0.5

            if a > Max:
                Max = a
            if a < Min:
                Min = a
    
    return count

if __name__ == '__main__':  # 测试kaze特征
    path1 = './static/images/lhy/001.jpg'
    path2 = './static/images/lhy/002.jpg'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    res = kaze_distance(img1, img2)
    res = round(res, 0)
    print(res)