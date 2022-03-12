import cv2

from cutimg import mycutimg

from gray_histogram_differential import main_gray_hist_differential

from main_color_demon import mymain_color

from edge_batch import main_edge
from edge_hist_batch import main_edge_hist_batch

from GLCM_demo import myGLCM_demo

from coner_demon import myconer

from blob_hist_correlation import myblobhist

'''
环境说明：
opencv要求3.4.1以下 新版本的opencv缺少相应的函数来运行coner_demon.py
'''

img_input = cv2.imread('static/images_GLCM_original/images_camouflage/mix/20m/1.JPG') # 红外 灰色
img_input2 = cv2.imread('static/images_GLCM_original/images_camouflage/mix/20m/2.JPG') # 红外 伪彩

# # cutimg.py 抠图 （有交互）
'''
输入：
img_input 格式为图片
输出:
path2：掩膜图像存储路径
path3：分割图像存储路径
'''
# mycutimg(img_input)


# # gray_histogram_differential.py 获取灰度直方图及其相关性
'''
输入：
无
输出：
直方图存储路径 path_mid
'''
path = 'static/images_GLCM_original'
path_bitwise = 'static/images_GLCM_bitwise'
path_gray_histogram_save = 'static/images_save/gray_histogram/'
# main_gray_hist_differential(path,path_bitwise,path_gray_histogram_save)


# # main_color_demo.py 获取彩色图像的主要颜色（在运行过程中会用到utils.py文件 而不是自带的utils包）
'''
输入:
img_input:图像
输出:
path2:主色提取结果存储路径
'''
# mymain_color(img_input2)


# # edge_batch.py 获取目标背景的边缘图像
'''
输入：
无
输出:
path3_edge_canny:存储边缘图像的路径
'''
# main_edge()

# none
# # edge_hist_batch.py 获取目标背景的边缘方向直方图及其相关性（必须要先运行edge_batch.py里的main_edge() 才能运行此程序）
'''
输入：
无输入
输出:
path3_edge_dh, everage_result:边缘方向直方图存储路径， 目标背景相关性结果
'''
# print(main_edge_hist_batch())


# # GLCM_demo.py 展示图像的多个GLCM可视化效果图
'''
输入：
image:图像
输出：
path2: GLCM可视化图像的存储路径
'''
# print(myGLCM_demo(img_input))


# # coner_demon.py
'''
输入：
无
输出：
path_save_coner：角点匹配情况图像存储路径
'''
# print(myconer())


# # blob_hist_correlation.py
'''
输入：
无
输出：
path_blob_hist_save：生成的斑块图像以及对应直方图存储路径
arr：result_area , result_length, result_circle, result_rect, result_stretch 几种形状特征的相似性
'''
print(myblobhist())