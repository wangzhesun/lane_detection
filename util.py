import cv2 as cv
import numpy as np


def sobel(img_channel, orient='x', sobel_kernel=3):
    sobel_img = []
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


def thresh_edge(hls_img, orig_img):
    _, l_thresh_img = cv.threshold(hls_img[:, :, 1], 120, 255, cv.THRESH_BINARY)
    l_thresh_img = cv.GaussianBlur(l_thresh_img, (3, 3), 0)

    l_thresh_img = mag_thresh(l_thresh_img, sobel_kernel=3, thresh=(110, 255))

    _, s_thresh_img = cv.threshold(hls_img[:, :, 2], 80, 255, cv.THRESH_BINARY)
    _, r_thresh_img = cv.threshold(orig_img[:, :, 2], 120, 255, cv.THRESH_BINARY)

    sr_thresh_img = cv.bitwise_and(s_thresh_img, r_thresh_img)
    srl_thresh_img = cv.bitwise_or(sr_thresh_img, l_thresh_img.astype(np.uint8))

    return srl_thresh_img


def transform_perspective(frame, roi, roi_transform):
    dim = frame.shape
    height = dim[0]
    width = dim[1]
    transform_matrix = cv.getPerspectiveTransform(roi, roi_transform)

    warped_img = cv.warpPerspective(frame, transform_matrix,
                                    [width, height], flags=cv.INTER_LINEAR)

    _, warped_img = cv.threshold(warped_img, 127, 255, cv.THRESH_BINARY)
    return warped_img
