import cv2

from config.setting import RESULT_FOLDER


# global img
# global point1, point2,point
# def on_mouse(event, x, y, flags, param):
#     global img, point1, point2,point,mode
#     img2 = img.copy()
#     if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
#         point = (x,y)
#         cv2.circle(img2, point, 10, (0,255,0), 5)
#         cv2.imshow('image', img2)
#         mode = False
#     if event == cv2.EVENT_LBUTTONDBLCLK:    #左键双击
#         point1 = (x, y)
#         cv2.circle(img2, point1, 10, (0, 255, 0), 4)
#         cv2.imshow('image', img2)
#         mode = True
#     elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON) and mode:               #按住左键拖曳
#         cv2.rectangle(img2, point1, (x,y), (255,0,0), 4)
#         cv2.imshow('image', img2)
#     elif event == cv2.EVENT_LBUTTONUP and mode:         #左键释放
#         point2 = (x,y)
#         cv2.rectangle(img2, point1, point2, (0,0,255), 4)
#         cv2.imshow('image', img2)
#         min_x = min(point1[0],point2[0])
#         min_y = min(point1[1],point2[1])
#         width = abs(point1[0] - point2[0])
#         height = abs(point1[1] -point2[1])
#         cut_img = img[min_y:min_y+height, min_x:min_x+width]
#         image_path = RESULT_FOLDER + '/SAR/image_input.png'
#         cv2.imwrite(image_path, cut_img)
#
# def get_image(path):
#     global img
#     img = cv2.imread(path)
#     cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback('image', on_mouse)
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     return 'done'

def get_image(points,path):
    if len(points) != 2:
        return 'The number of points is faulse'
    image = cv2.imread(path)
    if image.ndim > 2:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    point1 = points[0]
    point2 = points[1]
    min_x = min(point1[0], point2[0])
    min_y = min(point1[1],point2[1])
    width = abs(point1[0] - point2[0])
    height = abs(point1[1] -point2[1])
    cut_img = img[min_y:min_y+height, min_x:min_x+width]
    cut_img_path = RESULT_FOLDER + '/SAR/image_input.png'
    cv2.imwrite(cut_img_path, cut_img)
    return cut_img_path




