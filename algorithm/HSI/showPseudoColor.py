import cv2
import algorithm.HSI as hsi


def show_image(image_path, out_path):
    red_band = 76
    blue_band = 15
    green_band = 46
    image = hsi.load_data(image_path)
    img_r = image[:, :, red_band]
    img_g = image[:, :, green_band]

    img_b = image[:, :, blue_band]
    pseudo_image = cv2.merge([img_b, img_g, img_r])
    pseudo_image = pseudo_image * 255.0
    #  输出路径
    # out_path = "image_result/pseudo_image.jpg"

    cv2.imwrite(out_path, pseudo_image)
    '''cv2.imshow("test", pseudo_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return out_path
