import cv2 as cv
import numpy as np


def sobel(img_channel, orient='x', sobel_kernel=3):
    if orient == 'x':
        sobel_img = cv.Sobel(img_channel, cv.CV_64F, 1, 0, sobel_kernel)
    if orient == 'y':
        sobel_img = cv.Sobel(img_channel, cv.CV_64F, 0, 1, sobel_kernel)

    return sobel_img


def to_binary(array, thresh, value=0):
    if value == 0:
        ret = np.ones_like(array)
    else:
        ret = np.zeros_like(array)
        value = 1

    ret[(array >= thresh[0]) & (array <= thresh[1])] = value
    return ret


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    sobel_x = np.absolute(sobel(image, 'x', sobel_kernel))
    sobel_y = np.absolute(sobel(image, 'y', sobel_kernel))

    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return to_binary(mag, thresh)
