import cv2 as cv
import numpy as np


def sobel(img_channel, orient='x', sobel_kernel=3):
    """
    get sobel processed image in a specific direction

    :param img_channel: input image channel
    :param orient: direction to take sobel operation
    :param sobel_kernel: user provided sobel kernel size
    :return: image formed from either x or y sobel filters
    """
    sobel_img = []
    if orient == 'x':
        sobel_img = cv.Sobel(img_channel, cv.CV_64F, 1, 0, sobel_kernel)
    if orient == 'y':
        sobel_img = cv.Sobel(img_channel, cv.CV_64F, 0, 1, sobel_kernel)
    return sobel_img


def to_binary(array, thresh, value=0):
    """
    threshold the provided array into the target value (zero or one)

    :param array: array to be processed
    :param thresh: user-input closed-interval threshold
    :param value: target value (zero or one) to be threshold into
    :return: binary threshold array
    """
    if value == 0:
        ret = np.ones_like(array)
    else:
        ret = np.zeros_like(array)
        value = 1

    ret[(array >= thresh[0]) & (array <= thresh[1])] = value
    return ret


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    """
    get threshold combination of x- and y-direction sobel images

    :param image: user provided image
    :param sobel_kernel: sobel kernel size
    :param thresh: user-input closed-interval threshold
    :return: threshold combination of x- and y-direction sobel images
    """
    sobel_x = np.absolute(sobel(image, 'x', sobel_kernel))
    sobel_y = np.absolute(sobel(image, 'y', sobel_kernel))

    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return to_binary(mag, thresh)


def thresh_edge(hls_img, orig_img):
    """
    get threshold image after threshold in L channel, S channel, and R channel

    :param hls_img: image in HLS representation
    :param orig_img: original image
    :return: threshold image
    """
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
    """
    get the image transformed using the original and target coordinates

    :param frame: user-provided frame
    :param roi: points on the frame
    :param roi_transform: target position to be transformed into
    :param plot: flag indicating whether a plot of the operation result is needed.
                 Default value is False
    :return: a plot of the result after perspective transformation
    """
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
    """
    get the left and right peaks of the frame histogram

    :param frame: frame to be processed
    :return: left and right peaks of the histogram
    """
    histogram = np.sum(frame[int(frame.shape[0] / 2):, :], axis=0)

    mid_point = np.int(len(histogram) / 2)

    left_peak = np.argmax(histogram[:mid_point])
    right_peak = np.argmax(histogram[mid_point:]) + mid_point

    return left_peak, right_peak


def calculate_curvature(height, x_left, x_right, y_left, y_right, y_m_per_p, x_m_per_p):
    """
    get curve radius of left and right lanes

    :param height: frame height
    :param x_left: x pixel coordinates of the left lane
    :param x_right: x pixel coordinates of the right lane
    :param y_left: y pixel coordinates of the left lane
    :param y_right: y pixel coordinates of the right lane
    :param y_m_per_p: meter per pixel in y direction
    :param x_m_per_p: meter per pixel in x direction
    :return: curve radius of left and right lanes
    """
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


def calculate_center_offset(height, width, roi, roi_transform, left_lane, right_lane, x_m_per_p):
    """
    get the offset of the camera relative to the center of the lane

    :param height: frame height
    :param width: frame width
    :param roi: coordinates embodying region of interest
    :param roi_transform: target coordinates to be transformed by roi coordinates
    :param left_lane: best-fit curve function of the left lane
    :param right_lane: best-fit curve function of the right lane
    :param x_m_per_p: meter per pixel in x direction
    :return: center offset of the camera
    """
    # Assume the camera is centered in the image.
    # Get position of car in centimeters
    car_location = width / 2

    # Fine the x coordinate of the lane line bottom
    bottom_left = left_lane[0] * height ** 2 + left_lane[1] * height + left_lane[2]
    bottom_right = right_lane[0] * height ** 2 + right_lane[1] * height + right_lane[2]

    lane_center = (bottom_right - bottom_left) / 2 + bottom_left

    center_array = np.array([height, lane_center, 1]).reshape((-1, 1))

    transform_matrix = cv.getPerspectiveTransform(roi, roi_transform)

    real_lane_center = np.matmul(transform_matrix, center_array)
    real_lane_center[0] = real_lane_center[0] / real_lane_center[2]
    real_lane_center[1] = real_lane_center[1] / real_lane_center[2]
    real_lane_center[2] = 1

    return (car_location - real_lane_center[1][0]) * x_m_per_p * 100
