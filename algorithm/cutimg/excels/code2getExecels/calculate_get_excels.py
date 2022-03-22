import cv2

from main_color import get_main_color_excels

from gray_histogram import get_gray_histogram_excels

from GLCM import  get_GLCM_excels

from LBP import get_LBP_excels

from GGCM import get_GGCM_excels

from GLDS import get_GLDS_excels

from myTamura import get_Tamura_excels


def main(path_original_img, path_cutimg, path_bitwise, path_excels_save):
    # 主色
    get_main_color_excels(path_cutimg, path_excels_save)
    get_gray_histogram_excels(path_original_img, path_bitwise, path_excels_save)
    get_GLCM_excels(path_cutimg, path_excels_save)
    get_LBP_excels(path_cutimg, path_excels_save)
    get_GGCM_excels(path_cutimg, path_excels_save)
    get_GLDS_excels(path_cutimg, path_excels_save)
    get_Tamura_excels(path_cutimg, path_excels_save)
    return

# 原始图像存储路径
path_original_img = 'D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM_original'
# 分割结果存储路径
path_cutimg = 'D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM'
# 掩膜图像存储路径
path_bitwise = 'D:/Python/Python/WZ_GLDM/webNew2/static/images_GLCM_bitwise'
# excel表格存储路径
path_excels_save = 'D:/Python/Python/WZ_GLDM/webNew2/static/excels_save_3.21'


main(path_original_img, path_cutimg, path_bitwise, path_excels_save)




