# -*- coding: utf-8 -*-
import cv2


class Rect(object):
    def __init__(self, pt1=(0, 0), pt2=(0, 0)):
        self.tl = pt1
        self.br = pt2
        self.regularize()

    def regularize(self):
        """
        make sure tl = TopLeft point, br = BottomRight point
        """
        tl = (min(self.tl[0], self.br[0]), min(self.tl[1], self.br[1]))
        br = (max(self.tl[0], self.br[0]), max(self.tl[1], self.br[1]))
        self.tl = tl
        self.br = br

    def get_center(self):
        """
        get center point of Rect
        """
        center_x = (self.tl[0] + self.br[0]) // 2
        center_y = (self.tl[1] + self.br[1]) // 2
        return center_x, center_y

    def get_width(self):
        """
        get width of Rect
        """
        return abs(self.br[0] - self.tl[0])

    def get_height(self):
        """
        get height of Rect
        """
        return abs(self.br[1] - self.tl[1])

    def height_over_width(self):
        """
        ratio of height over width
        """
        return self.get_height() / self.get_width()

    def get_area(self):
        """
        get area of Rect
        """
        return self.get_width() * self.get_height()


class DrawZoom(object):
    def __init__(self, image, color,
                 current_pt=(0, 0),
                 default_zoom_image_size=(256, 256)):
        self.original_image = image
        self.color = color
        self.thickness = 2
        self.current_pt = current_pt
        self.default_zoom_image_size = default_zoom_image_size

        self.rect_in_big_image = Rect()
        self.rect_in_zoom_image = Rect()
        self.zoom_offset = (0, 0)

        self.is_drawing_big = False
        self.is_drawing_zoom = False
        self.exist_rect = False

        self.big_image = image.copy()
        self.zoom_image = None
        self.zoom_image_backup = None
        self.get_zoom_image()

    def get_image_height(self):
        """
        get height of big image
        """
        return self.original_image.shape[0]

    def get_image_width(self):
        """
        get width of big image
        """
        return self.original_image.shape[1]

    def get_margin_height(self):
        """
        get height of margin. in the margin area of big image, coordinate of
        current_pt does NOT change
        """
        return self.default_zoom_image_size[0] // 2

    def get_margin_width(self):
        """
        get width of margin
        """
        return self.default_zoom_image_size[1] // 2

    def get_zoom_image(self, height_ratio_expand=0.2, width_ratio_expand=0.2):
        """
        get zoom image for two cases: the rect exists or not.
        height_ratio_expand and width_ratio_expand are used for expanding some
        area of rect
        """
        if not self.exist_rect:
            self.get_zoom_image_for_current_pt()
        elif self.rect_in_big_image.get_area() > 0:
            self.get_zoom_image_for_rect(height_ratio_expand,
                                         width_ratio_expand)

    def get_zoom_image_for_current_pt(self):
        """
        get zoom image for current mouse point (when rect does not exist)
        """
        # (x, y) is center coordinate
        x = max(self.current_pt[0], self.get_margin_width())
        x = min(x, self.get_image_width() - self.get_margin_width())
        y = max(self.current_pt[1], self.get_margin_height())
        y = min(y, self.get_image_height() - self.get_margin_height())

        tl_x = x - self.get_margin_width()
        tl_y = y - self.get_margin_height()
        br_x = x + self.get_margin_width()
        br_y = y + self.get_margin_height()
        tl_x, tl_y, br_x, br_y = self.shrink_rect(tl_x, tl_y, br_x, br_y)
        self.zoom_image = self.big_image[tl_y:br_y, tl_x:br_x]
        self.zoom_image_backup = self.original_image[tl_y:br_y, tl_x:br_x]
        self.zoom_offset = (tl_x, tl_y)

    def get_zoom_image_for_rect(self, height_ratio_expand, width_ratio_expand):
        """
        get zoom image when rect exists
        """
        if self.rect_in_big_image.get_area() == 0:
            return None
        height_over_width_for_win_zoom = self.default_zoom_image_size[1] / self.default_zoom_image_size[0]
        center = self.rect_in_big_image.get_center()
        if self.rect_in_big_image.height_over_width() > height_over_width_for_win_zoom:
            half_height = int(0.5 * (1 + height_ratio_expand) *
                              self.rect_in_big_image.get_height())
            half_width = int(half_height / height_over_width_for_win_zoom)
        else:
            half_width = int(0.5 * (1 + width_ratio_expand) *
                             self.rect_in_big_image.get_width())
            half_height = int(half_width * height_over_width_for_win_zoom)
        tl_x = center[0] - half_width
        tl_y = center[1] - half_height
        br_x = center[0] + half_width
        br_y = center[1] + half_height
        tl_x, tl_y, br_x, br_y = self.shrink_rect(tl_x, tl_y, br_x, br_y)
        self.zoom_image = self.big_image[tl_y:br_y, tl_x:br_x]
        self.zoom_image_backup = self.original_image[tl_y:br_y, tl_x:br_x]
        self.zoom_offset = (tl_x, tl_y)

    @staticmethod
    def clip(value, low, high):
        """
        clip value between low and high
        """
        output = max(value, low)
        output = min(output, high)
        return output

    def shrink_point(self, x, y):
        """
        shrink point (x, y) to inside big image
        """
        x_shrink = self.clip(x, 0, self.get_image_width())
        y_shrink = self.clip(y, 0, self.get_image_height())
        return x_shrink, y_shrink

    def shrink_rect(self, pt1_x, pt1_y, pt2_x, pt2_y):
        """
        shrink rect to inside big image
        """
        pt1_x = self.clip(pt1_x, 0, self.get_image_width())
        pt1_y = self.clip(pt1_y, 0, self.get_image_height())
        pt2_x = self.clip(pt2_x, 0, self.get_image_width())
        pt2_y = self.clip(pt2_y, 0, self.get_image_height())
        rect = Rect((pt1_x, pt1_y), (pt2_x, pt2_y))
        rect.regularize()
        tl_x, tl_y = rect.tl
        br_x, br_y = rect.br
        return tl_x, tl_y, br_x, br_y

    def reset_big_image(self):
        """
        reset big_image (for show) using original image
        """
        self.big_image = self.original_image.copy()

    def reset_zoom_image(self):
        """
        reset zoom_image (for show) using the zoom image backup
        """
        self.zoom_image = self.zoom_image_backup.copy()

    def draw_rect_in_big_image(self):
        """
        draw rect in big image
        """
        cv2.rectangle(self.big_image,
                      self.rect_in_big_image.tl, self.rect_in_big_image.br,
                      color=self.color, thickness=self.thickness)

    def draw_rect_in_zoom_image(self):
        """
        draw rect in zoom image
        """
        cv2.rectangle(self.zoom_image,
                      self.rect_in_zoom_image.tl, self.rect_in_zoom_image.br,
                      color=self.color, thickness=self.thickness)

    def update_drawing_big(self):
        """
        update drawing big image, and map the corresponding area to zoom image
        """
        if self.exist_rect:
            self.draw_rect_in_big_image()
        self.get_zoom_image()

    def update_drawing_zoom(self):
        """
        update drawing big and zoom image when drawing rect in zoom image
        """
        if self.exist_rect:
            self.draw_rect_in_big_image()
            self.draw_rect_in_zoom_image()


def onmouse_big_image(event, x, y, flags, draw_zoom):
    if event == cv2.EVENT_LBUTTONDOWN:
        # pick first point of rect
        draw_zoom.is_drawing_big = True
        draw_zoom.rect_in_big_image.tl = (x, y)
        draw_zoom.exist_rect = True
    elif draw_zoom.is_drawing_big and event == cv2.EVENT_MOUSEMOVE:
        # pick second point of rect and draw current rect
        draw_zoom.rect_in_big_image.br = draw_zoom.shrink_point(x, y)
        draw_zoom.reset_big_image()
        draw_zoom.update_drawing_big()
    elif event == cv2.EVENT_LBUTTONUP:
        # finish drawing current rect
        draw_zoom.is_drawing_big = False
        draw_zoom.rect_in_big_image.br = draw_zoom.shrink_point(x, y)
        draw_zoom.rect_in_big_image.regularize()
        if draw_zoom.rect_in_big_image.get_area() == 0:
            draw_zoom.reset_big_image()
            draw_zoom.rect_in_big_image = Rect()
            draw_zoom.exist_rect = False
        draw_zoom.update_drawing_big()
    elif (not draw_zoom.is_drawing_big) and event == cv2.EVENT_RBUTTONDOWN:
        # right button down to erase current rect
        draw_zoom.rect_in_big_image = Rect()
        draw_zoom.exist_rect = False
        draw_zoom.reset_big_image()
        draw_zoom.update_drawing_big()
    else:
        # default case: mouse move without rect
        draw_zoom.current_pt = (x, y)
        draw_zoom.update_drawing_big()


def onmouse_zoom_image(event, x, y, flags, draw_zoom):
    if event == cv2.EVENT_LBUTTONDOWN:
        # pick first point of rect
        draw_zoom.is_drawing_zoom = True
        draw_zoom.rect_in_zoom_image.tl = (x, y)
        draw_zoom.rect_in_big_image.tl = (x + draw_zoom.zoom_offset[0],
                                          y + draw_zoom.zoom_offset[1])
        draw_zoom.exist_rect = True
    elif draw_zoom.is_drawing_zoom and event == cv2.EVENT_MOUSEMOVE:
        # pick second point of rect and draw current rect
        draw_zoom.rect_in_zoom_image.br = (x, y)
        draw_zoom.rect_in_big_image.br = draw_zoom.shrink_point(
            x + draw_zoom.zoom_offset[0], y + draw_zoom.zoom_offset[1])
        draw_zoom.reset_zoom_image()
        draw_zoom.reset_big_image()
        draw_zoom.update_drawing_zoom()
    elif event == cv2.EVENT_LBUTTONUP:
        # finish drawing current rect
        draw_zoom.is_drawing_zoom = False
        draw_zoom.rect_in_big_image.br = draw_zoom.shrink_point(
            x + draw_zoom.zoom_offset[0], y + draw_zoom.zoom_offset[1])
        draw_zoom.rect_in_big_image.regularize()
        if draw_zoom.rect_in_big_image.get_area() == 0:
            draw_zoom.reset_big_image()
            draw_zoom.rect_in_big_image = Rect()
            draw_zoom.rect_in_zoom_image = Rect()
            draw_zoom.exist_rect = False
            draw_zoom.current_pt = draw_zoom.shrink_point(
                x + draw_zoom.zoom_offset[0], y + draw_zoom.zoom_offset[1])
        draw_zoom.update_drawing_big()
    elif (not draw_zoom.is_drawing_big) and event == cv2.EVENT_RBUTTONDOWN:
        # right button down to erase current rect
        draw_zoom.rect_in_big_image = Rect()
        draw_zoom.rect_in_zoom_image = Rect()
        draw_zoom.exist_rect = False
        draw_zoom.reset_big_image()
        draw_zoom.current_pt = draw_zoom.shrink_point(
            x + draw_zoom.zoom_offset[0], y + draw_zoom.zoom_offset[1])
        draw_zoom.update_drawing_big()
    else:
        # mousemove in zoom image will not change the content of image
        pass


if __name__ == '__main__':
    WIN_NAME_BIG = 'big_image'
    WIN_NAME_ZOOM = 'zoom_image'
    image = cv2.imread(r'D:\sar\实验图像\成都图像\20211012成都24（4~44）调亮.tif')
    draw_zoom = DrawZoom(image, (0, 255, 0))
    cv2.namedWindow(WIN_NAME_BIG, 0)
    cv2.namedWindow(WIN_NAME_ZOOM, 0)
    cv2.setMouseCallback(WIN_NAME_BIG, onmouse_big_image, draw_zoom)
    cv2.setMouseCallback(WIN_NAME_ZOOM, onmouse_zoom_image, draw_zoom)
    while True:
        cv2.imshow(WIN_NAME_BIG, draw_zoom.big_image)
        cv2.imshow(WIN_NAME_ZOOM, draw_zoom.zoom_image)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()