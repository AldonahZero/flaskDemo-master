# /Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg2/static/excels_save/color_gray_mean/excel_color_gray_mean.xls
import os


def get_server_file_path(abs_file_path):
    return abs_file_path.split('flaskDemo-master')[1][1:]

if __name__ == '__main__':
    print(get_server_file_path('/Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg2/static/excels_save/color_gray_mean/excel_color_gray_mean.xls'))
    print(get_server_file_path(
        '/Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg2/static/excels_save/color_gray_mean\\excel_color_gray_mean.xls'))