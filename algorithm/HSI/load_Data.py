import numpy as np


def load_data(image_path):
    raw_image = np.fromfile(image_path, "float32")

    image = np.zeros((1057, 960, 176))
    for row in range(0, 1057):
        for dim in range(0, 176):
            image[row, :, dim] = raw_image[(dim + row * 176) * 960:(dim + 1 + row * 176) * 960]
    return image


