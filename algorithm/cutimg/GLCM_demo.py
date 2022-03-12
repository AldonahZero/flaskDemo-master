# coding: utf-8
# The code is written by Linghui

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import data
from math import floor, ceil
from skimage.feature import greycomatrix, greycoprops
import os
import shutil

'''
输入：
number=1, window_size=7, stride=2, angle_get=0，参数固定
path1: 原始图像的输入路径
path2: 生成的GLCM可视化图像存储路径
输出：
将生成的可视化图像存储于path2
'''
def main():
    pass


def image_patch(img2, slide_window, h, w):
    image = img2
    window_size = slide_window
    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)

    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            patch[:, :, i, j] = img2[i: i + slide_window, j: j + slide_window]

    return patch


def calcu_glcm(img, vmin=0, vmax=255, nbit=64, slide_window=5, step=[2], angle=[0]):
    mi, ma = vmin, vmax
    h, w = img.shape

    # Compressed gray range：vmin: 0-->0, vmax: 256-1 -->nbit-1
    bins = np.linspace(mi, ma + 1, nbit + 1)
    img1 = np.digitize(img, bins) - 1

    # (512, 512) --> (slide_window, slide_window, 512, 512)
    img2 = cv2.copyMakeBorder(img1, floor(slide_window / 2), floor(slide_window / 2)
                              , floor(slide_window / 2), floor(slide_window / 2), cv2.BORDER_REPLICATE)  # 图像扩充

    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)
    patch = image_patch(img2, slide_window, h, w)

    # Calculate GLCM (5, 5, 512, 512) --> (64, 64, 512, 512)
    # greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
    glcm = np.zeros((nbit, nbit, len(step), len(angle), h, w), dtype=np.uint8)
    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            glcm[:, :, :, :, i, j] = greycomatrix(patch[:, :, i, j], step, angle, levels=nbit)

    return glcm


def calcu_glcm_mean(glcm, nbit=64):
    '''
    calc glcm mean
    '''
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    return mean


def calcu_glcm_variance(glcm, nbit=64):
    '''
    calc glcm variance
    '''
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    variance = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean) ** 2

    return variance


def calcu_glcm_homogeneity(glcm, nbit=64):
    '''
    calc glcm Homogeneity
    '''
    Homogeneity = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            Homogeneity += glcm[i, j] / (1. + (i - j) ** 2)

    return Homogeneity


def calcu_glcm_contrast(glcm, nbit=64):
    '''
    calc glcm contrast
    '''
    contrast = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            contrast += glcm[i, j] * (i - j) ** 2

    return contrast


def calcu_glcm_dissimilarity(glcm, nbit=64):
    '''
    calc glcm dissimilarity
    '''
    dissimilarity = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            dissimilarity += glcm[i, j] * np.abs(i - j)

    return dissimilarity


def calcu_glcm_entropy(glcm, nbit=64):
    '''
    calc glcm entropy
    '''
    eps = 0.00001
    entropy = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            entropy -= glcm[i, j] * np.log10(glcm[i, j] + eps)

    return entropy


def calcu_glcm_energy(glcm, nbit=64):
    '''
    calc glcm energy or second moment
    '''
    energy = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            energy += glcm[i, j] ** 2

    return energy


def calcu_glcm_correlation(glcm, nbit=64):
    '''
    calc glcm correlation (Unverified result)
    '''

    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    variance = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean) ** 2

    correlation = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            correlation += ((i - mean) * (j - mean) * (glcm[i, j] ** 2)) / variance

    return correlation


def calcu_glcm_Auto_correlation(glcm, nbit=64):
    '''
    calc glcm auto correlation
    '''
    Auto_correlation = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            Auto_correlation += glcm[i, j] * i * j

    return Auto_correlation


def myGLCM_demo(image_url):
    main()
    mymain_color_real_path_prex = os.path.dirname(os.path.realpath(__file__))
    mymain_color_real_path = os.path.join(mymain_color_real_path_prex,'static/images_save/GLCM_demo')

    if (len(os.listdir(mymain_color_real_path)) != 0):
        # GLCM可视化图存储文件夹
        shutil.rmtree(mymain_color_real_path)
        os.mkdir(mymain_color_real_path)

    image = cv2.imread(image_url)
    number = 1
    window_size = 7
    stride = 2
    angle_get = 0
    # path1 = 'static/images_GLCM_original/images_camouflage/mix/20m/'
    path2 = mymain_color_real_path

    # start = time.time()

    # print('---------------0. Parameter Setting-----------------')
    nbit = 64  # gray levels
    mi, ma = 0, 255  # max gray and min gray
    slide_window = window_size  # sliding window
    # step = [2, 4, 8, 16] # step
    # angle = [0, np.pi/4, np.pi/2, np.pi*3/4] # angle or direction
    step = [stride]
    angle = [angle_get]
    # print('-------------------1. Load Data---------------------')
    # image = 'static\\images_GLCM_original\\images_camouflage\\mix\\20m\\' + str(number) + '.JPG'
    # image = path1 + str(number) + '.JPG'
    # img_input = np.array(Image.open(image))  # If the image has multi-bands, it needs to be converted to grayscale image
    img_input = np.array(image)
    img_input = np.uint8(255.0 * (img_input - np.min(img_input)) / (np.max(img_input) - np.min(img_input)))  # normalization
    if img_input.ndim == 3:
        img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    elif img_input.ndim == 1:
        img = img_input
    else:
        return 'Please input correct picture!'
    h, w = img.shape
    # print('------------------2. Calcu GLCM---------------------')
    glcm = calcu_glcm(img, mi, ma, nbit, slide_window, step, angle)
    # print('-----------------3. Calcu Feature-------------------')
    #
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = np.zeros((nbit, nbit, h, w), dtype=np.float32)
            glcm_cut = glcm[:, :, i, j, :, :]
            # mean = calcu_glcm_mean(glcm_cut, nbit)
            # variance = calcu_glcm_variance(glcm_cut, nbit)
            homogeneity = calcu_glcm_homogeneity(glcm_cut, nbit)
            contrast = calcu_glcm_contrast(glcm_cut, nbit)
            dissimilarity = calcu_glcm_dissimilarity(glcm_cut, nbit)
            entropy = calcu_glcm_entropy(glcm_cut, nbit)
            energy = calcu_glcm_energy(glcm_cut, nbit)
            correlation = calcu_glcm_correlation(glcm_cut, nbit)
            # Auto_correlation = calcu_glcm_Auto_correlation(glcm_cut, nbit)
    # print('---------------4. Display and Result----------------')
    plt.figure(figsize=(10, 4.5))
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 12,
            }

    plt.subplot(2, 4, 1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.title('Original', font)

    # plt.subplot(2, 5, 2)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(mean, cmap='gray')
    # plt.title('Mean', font)

    # plt.subplot(2, 5, 3)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(variance, cmap='gray')
    # plt.title('Variance', font)

    plt.subplot(2, 4, 2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(homogeneity, cmap='gray')
    plt.title('Homogeneity', font)

    plt.subplot(2, 4, 3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(contrast, cmap='gray')
    plt.title('Contrast', font)

    plt.subplot(2, 4, 4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(dissimilarity, cmap='gray')
    plt.title('Dissimilarity', font)

    plt.subplot(2, 4, 6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(entropy, cmap='gray')
    plt.title('Entropy', font)

    plt.subplot(2, 4, 7)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(energy, cmap='gray')
    plt.title('Energy', font)

    plt.subplot(2, 4, 8)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(correlation, cmap='gray')
    plt.title('Correlation', font)

    # plt.subplot(2, 5, 10)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(Auto_correlation, cmap='gray')
    # plt.title('Auto Correlation', font)

    plt.tight_layout(pad=0.5)
    # plt.savefig('static\\images_save\\GLCM_demo\\GLCM_Features.png'
    plt.savefig(path2 + '/GLCM_Features.png'
                , format='png'
                , bbox_inches='tight'
                , pad_inches=0
                , dpi=300)
    # plt.show()

    # end = time.time()
    # print('Code run time:', end - start)
    return path2

# myGLCM_demo(number=1, window_size=7, stride=2, angle_get=0, path1=path1, path2=path2)

