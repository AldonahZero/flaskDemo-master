# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import uuid

import cv2
import os
import piexif
from PIL import Image
from numpy.core.defchararray import rfind

RESIZE_PIC_PATH = "algorithm/mosaic/static/resizeImg/"
ORIGIN_PIC_PATH = "algorithm/mosaic/static/images/"
RESULT_PATH = "algorithm/mosaic/static/result/"
MOSAIC_EXE_PATH = "algorithm\\mosaic\\mosaicing\\mosaicing.exe"


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def resize(path):
    myList = os.listdir(path)
    i = 0
    folder_name = path.split("/")[-1]
    if not os.path.exists(RESIZE_PIC_PATH + folder_name):
        os.makedirs(RESIZE_PIC_PATH + folder_name)
    for Img in myList:
        curImg = cv2.imread(f'{path}/{Img}')
        img = Image.open(f'{path}/{Img}')
        exif_dict = piexif.load(img.info['exif'])

        curImg1 = img.resize((int(curImg.shape[1] * 0.2), int(curImg.shape[0] * 0.2)),
                             Image.ANTIALIAS)

        exif_bytes = piexif.dump(exif_dict)
        curImg1.save(RESIZE_PIC_PATH + folder_name + '/image' + str(i) + '.jpg', exif=exif_bytes)
        i = i + 1;


def mosaic(file_name):
    out_path = RESULT_PATH + file_name
    # 调整图片大小和名字
    resize(ORIGIN_PIC_PATH + file_name)
    out_file_name = str(uuid.uuid1())
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    path_01 = MOSAIC_EXE_PATH + r" %s %s %s" % (RESIZE_PIC_PATH + file_name, out_path, out_file_name)
    r_v = os.system(path_01)
    return out_file_name


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # path_01 = r"mosaicing\\mosaicing.exe %s %s" % (r"mosaicing\\test_data", r"mosaicing\\result")
    # r_v = os.system(path_01)
    # print(r_v)
    mosaic("2a4ebbb4-b299-11ec-94ce-38fc98179f93")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
