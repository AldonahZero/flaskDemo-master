# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import os


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folder = 'x'
    path = r'E:\CHENGXU\pinjie\resizepic\c'
    images = []
    myList = os.listdir(path)
    i = 0
    for Img in myList:
        curImg = cv2.imread(f'{path}/{Img}')
        print(i)
        curImg1 = cv2.resize(curImg, (int(curImg.shape[1] * 0.2), int(curImg.shape[0] * 0.2)),
                            interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path+'\image'+str(i)+'.jpg',curImg1)
        i = i + 1;
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
