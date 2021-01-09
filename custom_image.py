import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread(r'C:\Users\User\Downloads\Untitled.png')


def convert_rgb_greyscale(img):
    """
    :param img: list
        array that contains [r, g, b] values
    :return:  converted_img

    """
    converted_img = []
    for row in img:
        new_row = []
        for rgb_values in row:
            new_row.append((0.3 * (1 - rgb_values[0])) + (0.59 * (1 - rgb_values[1])) + (0.11 * (1 - rgb_values[2])))
        converted_img.append(new_row)
    return np.reshape(converted_img, (28,28))


def reshape_data(pic, actual):
    """

    :param pic: list
        28x28 array
    :param actual:
    :return:
    """
    new_data = np.reshape(pic, (784, 1))
    return new_data, actual
