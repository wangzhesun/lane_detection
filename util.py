import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


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
    l_mean = np.percentile(hls_img[:, :, 1], 85)
    s_mean = np.percentile(hls_img[:, :, 2], 85)
    r_mean = np.percentile(orig_img[:, :, 2], 85)
    _, l_thresh_img = cv.threshold(hls_img[:, :, 1], l_mean, 255, cv.THRESH_BINARY)  # 120
    l_thresh_img = cv.GaussianBlur(l_thresh_img, (3, 3), 0)

    l_thresh_img = mag_thresh(l_thresh_img, sobel_kernel=3, thresh=(110, 255))

    _, s_thresh_img = cv.threshold(hls_img[:, :, 2], s_mean, 255, cv.THRESH_BINARY)  # 80
    _, r_thresh_img = cv.threshold(orig_img[:, :, 2], r_mean, 255, cv.THRESH_BINARY)  # 120

    sr_thresh_img = cv.bitwise_and(s_thresh_img, r_thresh_img)

    srl_thresh_img = cv.bitwise_or(sr_thresh_img, l_thresh_img.astype(np.uint8))

    return srl_thresh_img


def transform_perspective(frame, roi, roi_transform, plot=False):
    height = frame.shape[0]
    width = frame.shape[1]
    transform_matrix = cv.getPerspectiveTransform(roi, roi_transform)

    warped_img = cv.warpPerspective(frame, transform_matrix,
                                    [width, height], flags=cv.INTER_LINEAR)

    if plot:
        cv.imshow('Perspective Transformed Image', warped_img)
        cv.waitKey(0)
    return warped_img


def calculate_histogram_peak(frame):
    histogram = np.sum(frame[int(frame.shape[0] / 2):, :], axis=0)

    mid_point = np.int(len(histogram) / 2)

    left_peak = np.argmax(histogram[:mid_point])
    right_peak = np.argmax(histogram[mid_point:]) + mid_point

    return left_peak, right_peak


def calculate_curvature(height, x_left, x_right, y_left, y_right, y_m_per_p, x_m_per_p):
    # Set the y-value where we want to calculate the road curvature.
    # Select the maximum y-value, which is the bottom of the frame.
    y_eval = height

    # Fit polynomial curves to the real world environment
    left_fit_cr = np.polyfit(y_left * y_m_per_p, x_left * x_m_per_p, 2)
    right_fit_cr = np.polyfit(y_right * y_m_per_p, x_right * x_m_per_p, 2)

    # Calculate the radii of curvature
    left_curve = ((1 + (2 * left_fit_cr[0] * y_eval * y_m_per_p + left_fit_cr[1]) ** 2) ** 1.5) \
                 / np.absolute(2 * left_fit_cr[0])
    right_curve = ((1 + (2 * right_fit_cr[0] * y_eval * x_m_per_p + right_fit_cr[1]) ** 2) ** 1.5) \
                  / np.absolute(2 * right_fit_cr[0])

    return left_curve, right_curve
