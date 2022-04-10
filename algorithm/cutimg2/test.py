import cv2
from .cutimg import mycutimg
from .color_gray_mean import myGrayMean
from .color_gray_mean_excelSave import myGrayMean_excelSave
from .color_gray_histogram import myGrayHitogram
from .color_gray_histogram_excelSave import myGrayHitogram_excelSave
from .color_main_color import myMainColor
from .color_main_color_excelSave import myMainColor_excelSave
from .edge import myEdge
from .edge_histogram import myEdgeHistogram
from .edge2 import myEdgeCanny
from .edge2 import myEdgeLaplacian
from .edge2 import myEdgeSobel
from .edge2 import myEdgeRoberts
from .edge2 import myEdgePrewitt
from .edge2 import myEdgeLog
from .edge_histogram2 import myEdgeHistogramCanny
from .edge_histogram2 import myEdgeHistogramLaplacian
from .edge_histogram2 import myEdgeHistogramSobel
from .edge_histogram2 import myEdgeHistogramRoberts
from .edge_histogram2 import myEdgeHistogramPrewitt
from .edge_histogram2 import myEdgeHistogramLog
from .texture_GLCM import myGLCM
from .texture_GLCM_excelSave import myGLCM_excelSave
from .texture_GGCM import myGGCM
from .texture_GGCM_excelSave import myGGCM_excelSave
from .texture_GLDS import myGLDS
from .texture_GLDS_excelSave import myGLDS_excelSave
from .texture_Tamura import myTamura
from .texture_Tamura_excelSave import myTamura_excelSave
from .texture_LBP_excelSave import myLBP_excelSave
from .Blob_Kmeans import myBlob_excelSave
from .coner_coner import myConer
from .coner_coner_excelsSave import myConer_excelSave


path_original = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_original/1.jpg'
img_input = cv2.imread(path_original) # 输入图像

path_bitwise = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_bitwise/1.jpg'  # 掩膜保存路径
path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_cutimg/'  # 分割结果保存路径

path_mainColor = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_mainColor/' # 主色提取结果存储路径

path_edge = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/' # 边缘保存路径
path_edge_canny = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/canny/' # 边缘保存路径
path_edge_laplacian = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/laplacian/' # 边缘保存路径
path_edge_log = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/log/' # 边缘保存路径
path_edge_prewitt = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/prewitt/' # 边缘保存路径
path_edge_roberts = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/roberts/' # 边缘保存路径
path_edge_sobel = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge/sobel/' # 边缘保存路径

path_edge_histogram = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge_histogram/' # 边缘方向直方图保存路径
path_edge_histogram_canny = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge_histogram/canny/' # 边缘方向直方图保存路径
path_edge_histogram_laplacian = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge_histogram/laplacian/' # 边缘方向直方图保存路径
path_edge_histogram_log = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge_histogram/log/' # 边缘方向直方图保存路径
path_edge_histogram_prewitt = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge_histogram/prewitt/' # 边缘方向直方图保存路径
path_edge_histogram_roberts = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge_histogram/roberts/' # 边缘方向直方图保存路径
path_edge_histogram_sobel = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_edge_histogram/sobel/' # 边缘方向直方图保存路径

path_blob_hist_save = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_blob/' #生成的斑块图像与统计直方图存储路径

path_coner = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/'
path_coner_ORB = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/ORB/'
path_coner_FAST = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/FAST/'
path_coner_SURF = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/SURF/'
path_coner_SIFT = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/SIFT/'
path_coner_BRISKF = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/BRISKF/'
path_coner_KAZE = 'D:/Python/Python/WZ_GLDM/webNew3/static/img_save_coner/KAZE/'


excels_color_gray_mean = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/color_gray_mean/' # 灰度均值表格存储路径
excels_color_gray_histogram = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/color_gray_histogram/' # 灰度均值表格存储路径
excels_color_main_color = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/color_main_color/' # 灰度均值表格存储路径

excels_edge_histogram = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/edge_histogram/' # 边缘方向直方图表格存储路径

excels_texture_GLCM = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/texture_GLCM/' # GLCM表格存储路径
excels_texture_GGCM = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/texture_GGCM/' # GGCM表格存储路径
excels_texture_GLDS = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/texture_GLDS/' # GLDS表格存储路径
excels_texture_Tamura = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/texture_Tamura/' # Tamura表格存储路径
excels_texture_LBP = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/texture_LBP/' # LBP表格存储路径

excels_blob_Kmeans = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/blob_Kmeans/' # Blob表格存储路径

excels_coner_coner = 'D:/Python/Python/WZ_GLDM/webNew3/static/excels_save/coner_coner/' # coner表格存储路径


# ---------------------------------------------图像分割预处理---------------------------------------------------
# cuimg.py 图像分割
'''
mycutimg(img_input)
输入为一张图片
生成的掩膜图像保存在 path_bitwise
分割后的图像保存在  path_cutimg
'''
# mycutimg(img_input)
#---------------------------------------------颜色特征--------------------------------------------------------
# color_gray_mean.py 颜色特征-灰度均值 可视化展示部分
'''
myGrayMean()
输入为 path_cutimg
算法处理的是path_cutimg中的所有图片，最后返回一个长度为9的数组arr
'''
# print(myGrayMean(path_cutimg))

# color_gray_mean_excelSave.py 颜色特征-灰度均值 表格存储
'''
myGrayMean_excelSave()
输入为 path_cutimg, excels_color_gray_mean
算法处理的是path_cutimg中的所有图片，最后返回一个路径里面存储到是灰度均值的excel表格
'''
# print(myGrayMean_excelSave(path_cutimg, excels_color_gray_mean))


# color_gray_histogram.py 颜色特征-灰度直方图 可视化展示部分
'''
myGrayHitogram()
输入为 path_original, path_bitwise
算法处理的是原图与path_bitwise对应的掩膜图像，返回一个长度为256的一维数组
'''
# print(myGrayHitogram(path_original, path_bitwise))

# color_gray_histogram_excelSave.py 颜色特征-灰度直方图 表格存储
'''
myGrayHitogram_excelSave()
输入为 path_original, path_bitwise, excels_color_gray_histogram
算法处理的是原图与path_bitwise对应的掩膜图像，表格的存储路径
'''
# print(myGrayHitogram_excelSave(path_original, path_bitwise, excels_color_gray_histogram))


# color_main_color.py 颜色特征-主色提取 可视化展示部分
'''
myMainColor()
输入为 path_cutimg, path_mainColor
算法处理的是path_cutimg中的目标图像14.jpg，返回值是主色提取结果展示的存储路径
'''
# print(myMainColor(path_cutimg, path_mainColor))

# color_main_color_excelSave.py 颜色特征-主色提取 表格存储部分
'''
myMainColor_excelSave()
输入为 path_cutimg, excels_color_main_color
算法处理的是path_cutimg中的九张图片，返回值是excel存储路径excels_color_main_color
'''
# print(myMainColor_excelSave(path_cutimg, excels_color_main_color))

#---------------------------------------------边缘特征--------------------------------------------------------
# edge.py 边缘特征-获取边缘 可视化展示部分（无需存储表格）
'''
myEdge()
输入为 path_cutimg, path_edge, path_edge_canny, path_edge_laplacian, path_edge_log, path_edge_prewitt, 
           path_edge_roberts, path_edge_sobel
算法处理的是path_cutimg中的九张图片，返回值是六种边缘图片文件夹的上层文件夹路径path_edge
'''
# print(myEdge(path_cutimg, path_edge, path_edge_canny, path_edge_laplacian, path_edge_log, path_edge_prewitt,
#            path_edge_roberts, path_edge_sobel))

# edge_histogram.py 后端生成图片并存储起来 但是没有要求可视化 + 表格存储部分
'''
myEdgeHistogram()
输入为 path_cutimg, path_edge, path_edge_canny, path_edge_laplacian, path_edge_log, path_edge_prewitt, 
                    path_edge_roberts, path_edge_sobel, path_edge_histogram, path_edge_histogram_canny, 
                    path_edge_histogram_laplacian, path_edge_histogram_log, path_edge_histogram_prewitt, 
                    path_edge_histogram_roberts, path_edge_histogram_sobel, excels_edge_histogram
算法处理的是path_cutimg中的九张图片，path_edge文件夹下六个文件夹里面的图片，
边缘方向直方图保存在path_edge_histogram，excel表保存在excels_edge_histogram
'''
# print(myEdgeHistogram(path_cutimg, path_edge, path_edge_canny, path_edge_laplacian, path_edge_log, path_edge_prewitt,
#                     path_edge_roberts, path_edge_sobel, path_edge_histogram, path_edge_histogram_canny,
#                     path_edge_histogram_laplacian, path_edge_histogram_log, path_edge_histogram_prewitt,
#                     path_edge_histogram_roberts, path_edge_histogram_sobel, excels_edge_histogram))

#=========================================边缘特征分开封装版本========
# edge2.py 多种算子分开封装
'''
输入为 原始子图路径path_cutimg、边缘图像文件路径path_edge、边缘图像子文件路径path_edge_canny（以及其他算子的路径）
返回值为 各种算子计算出来的边缘图像的存储路径
'''
# myEdgeCanny(path_cutimg, path_edge, path_edge_canny)
# myEdgeLaplacian(path_cutimg, path_edge, path_edge_laplacian)
# myEdgeLog(path_cutimg, path_edge, path_edge_log)
# myEdgeRoberts(path_cutimg, path_edge, path_edge_roberts)
# myEdgePrewitt(path_cutimg, path_edge, path_edge_prewitt)
# myEdgeSobel(path_cutimg, path_edge, path_edge_sobel)

# edge_histogram2.py 多种算子分开封装
'''
输入为 边缘图像路径path_edge_canny、边缘方向直方图存储路径path_edge_histogram_canny、表格存储路径excels_edge_histogram
返回值为 边缘方向直方图存储路径path_edge_histogram_canny、表格存储路径excels_edge_histogram
'''
# myEdgeHistogramCanny(path_edge_canny, path_edge_histogram_canny, excels_edge_histogram)
# myEdgeHistogramLaplacian(path_edge_laplacian, path_edge_histogram_laplacian, excels_edge_histogram)
# myEdgeHistogramLog(path_edge_log, path_edge_histogram_log, excels_edge_histogram)
# myEdgeHistogramSobel(path_edge_sobel, path_edge_histogram_sobel, excels_edge_histogram)
# myEdgeHistogramPrewitt(path_edge_prewitt, path_edge_histogram_prewitt, excels_edge_histogram)
# myEdgeHistogramRoberts(path_edge_roberts, path_edge_histogram_roberts, excels_edge_histogram)
#---------------------------------------------纹理特征--------------------------------------------------------
# texture_GLCM.py  可视化部分
'''
myGLCM()
输入为 path_cutimg
算法处理的是path_cutimg中的目标图片14.jpg
返回一个长度为6的一维矩阵 元素值分别代表 对比度、同质性、能量、相关性、角二阶矩、差异性
'''
# print(myGLCM(path_cutimg))

# texture_GLCM_excelSave.py  表格存储部分
'''
myGLCM_excelSave()
输入为 path_cutimg, excels_texture_GLCM
算法处理的是path_cutimg中的九张图片
返回一个路径excels_texture_GLCM 存储GLCM值的excel表
'''
# print(myGLCM_excelSave(path_cutimg, excels_texture_GLCM))


# texture_GGCM.py  可视化部分
'''
myGGCM()
输入为 path_cutimg
算法处理的是path_cutimg中的目标图片14.jpg
返回一个长度为15的一维矩阵 元素值分别代表 小梯度优势、大梯度优势、灰度不均匀性、梯度不均匀性、能量、灰度均值、梯度均值、灰度均方差、梯度均方差、
相关性、灰度熵、梯度熵、混合熵、惯性、逆差矩
'''
# print(myGGCM(path_cutimg))

# texture_GGCM_excelSave.py  表格存储部分
'''
myGGCM_excelSave()
输入为 path_cutimg, excels_texture_GGCM
算法处理的是path_cutimg中的九张图片
返回一个路径excels_texture_GGCM 存储GGCM值的excel表
'''
# print(myGGCM_excelSave(path_cutimg, excels_texture_GGCM))


# texture_GLDS.py  可视化部分
'''
myGLDS()
输入为 path_cutimg
算法处理的是path_cutimg中的目标图片14.jpg
返回一个长度为4的一维矩阵，元素值分别代表 均值、对比度、角二阶矩、熵
'''
# print(myGLDS(path_cutimg))

# texture_GLDS_excelSave.py  表格存储部分
'''
myGLDS_excelSave()
输入为 path_cutimg, excels_texture_GLDS
算法处理的是path_cutimg中的九张图片
返回一个路径excels_texture_GLDS 存储GLDS值的excel表
'''
# print(myGLDS_excelSave(path_cutimg, excels_texture_GLDS))


# texture_Tamura.py  可视化部分
'''
myTamura()
输入为 path_cutimg
算法处理的是path_cutimg中的目标图片14.jpg
返回一个长度为4的一维矩阵，元素值分别代表 粗糙度、对比度、方向度、线性度
'''
# print(myTamura(path_cutimg))

# texture_Tamura_excelSave.py  表格存储部分
'''
myTamura_excelSave()
输入为 path_cutimg, excels_texture_Tamura
算法处理的是path_cutimg中的九张图片
返回一个路径excels_texture_Tamura 存储Tamura值的excel表
'''
# print(myTamura_excelSave(path_cutimg, excels_texture_Tamura))


# texture_LBP_excelSave.py 表格存储部分（无可视化）
'''
myLBP_excelSave()
输入为 path_cutimg, excels_texture_LBP
算法处理的是path_cutimg中的9张图片
返回一个路径excels_texture_LBP 存储目标背景LBP相关性值的excel表
'''
# print(myLBP_excelSave(path_cutimg, excels_texture_LBP))


#---------------------------------------------斑块特征--------------------------------------------------------
# Blob_Kmeans.py  可视化与表格存储同时运行
''' 
myBlob_excelSave()
输入为 path_cutimg, excels_blob_Kmeans, path_blob_hist_save
算法处理的是path_cutimg中的9张图片
返回一个表格存储路径excels_blob_Kmeans
'''
# print(myBlob_excelSave(path_cutimg, excels_blob_Kmeans, path_blob_hist_save))


#---------------------------------------------角点特征--------------------------------------------------------
# coner_coner.py 可视化部分
'''
myConer()
输入为 path_cutimg, path_coner, path_coner_ORB, path_coner_FAST, path_coner_SURF, path_coner_SIFT, 
            path_coner_BRISKF, path_coner_KAZE
算法处理的是path_cutimg中的目标图片14.jpg
返回图片存储路径path_coner
'''
# print(myConer(path_cutimg, path_coner, path_coner_ORB, path_coner_FAST, path_coner_SURF, path_coner_SIFT,
#             path_coner_BRISKF, path_coner_KAZE))

# coner_coner_excelSave.py 表格存储部分
'''
myConer_excelSave()
输入为 path_cutimg, excels_coner_coner
算法处理的是path_cutimg中的九张图片
返回表格存储路径excels_coner_coner
'''
# print(myConer_excelSave(path_cutimg, excels_coner_coner))




