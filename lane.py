import cv2 as cv
import numpy as np
import util
from matplotlib import pyplot as plt


class Lane:
    def __init__(self):
        self.mode_ = 'image'
        self.lane_file_ = []
        self.roi_select_img_ = []
        self.roi_ = []
        self.roi_transform_ = []
        self.window_height_ = 30
        self.height_ = -1
        self.width_ = -1
        self.min_pixel_recenter_ = -1
        self.margin_ = -1
        self.left_lane_ = []
        self.right_lane_ = []

    def set_mode(self, mode):
        self.mode_ = mode

    def load_lane(self, file_path):
        if self.mode_ == 'image':
            try:
                self.lane_file_ = cv.imread(file_path)
            except FileNotFoundError:
                print("No file or directory with the name {}".format(file_path))
        else:
            self.lane_file_ = cv.VideoCapture(file_path)
            assert self.lane_file_.isOpened(), 'Cannot capture source'

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            self.roi_.append([x, y])
            print(x, ' ', y)

            self.roi_select_img_ = cv.circle(self.roi_select_img_, (x, y), radius=5,
                                             color=(255, 0, 0),
                                             thickness=-1)
            cv.imshow('image', self.roi_select_img_)

    def set_roi(self, frame):
        cv.imshow('image', frame)
        cv.setMouseCallback('image', self.click_event)
        cv.waitKey(0)

        self.roi_[1][1] = self.height_
        self.roi_[2][1] = self.height_
        self.roi_ = np.float32(self.roi_)

        self.roi_transform_.append([0.15 * self.width_, 0])
        self.roi_transform_.append([0.15 * self.width_, self.height_])
        self.roi_transform_.append([0.85 * self.width_, self.height_])
        self.roi_transform_.append([0.85 * self.width_, 0])
        self.roi_transform_ = np.float32(self.roi_transform_)

    def detect_lane_pixels(self, frame, plot=False):
        window_num = int(self.height_ / self.window_height_)

        non_zero_pixels = np.nonzero(frame)
        non_zero_pixels_y = non_zero_pixels[0]
        non_zero_pixels_x = non_zero_pixels[1]

        left_lane_pixels_id = []
        right_lane_pixels_id = []
        frame_sliding_window = frame.copy()

        x_left_center, x_right_center = util.calculate_histogram_peak(frame)

        for window in range(window_num):
            window_y_low = self.height_ - window * self.window_height_
            window_y_high = self.height_ - (window + 1) * self.window_height_
            window_x_left_low = x_left_center - self.margin_
            window_x_left_high = x_left_center + self.margin_
            window_x_right_low = x_right_center - self.margin_
            window_x_right_high = x_right_center + self.margin_

            if plot:
                cv.rectangle(frame_sliding_window, (int(window_x_left_low), int(window_y_low)), (
                    int(window_x_left_high), int(window_y_high)), (255, 255, 255), 2)
                cv.rectangle(frame_sliding_window, (int(window_x_right_low), int(window_y_low)), (
                    int(window_x_right_high), int(window_y_high)), (255, 255, 255), 2)

            good_left_pixel_id = ((non_zero_pixels_y < window_y_low) &
                                  (non_zero_pixels_y > window_y_high) &
                                  (non_zero_pixels_x > window_x_left_low) &
                                  (non_zero_pixels_x < window_x_left_high)).nonzero()[0]
            good_right_pixel_id = ((non_zero_pixels_y < window_y_low) &
                                   (non_zero_pixels_y > window_y_high) &
                                   (non_zero_pixels_x > window_x_right_low) &
                                   (non_zero_pixels_x < window_x_right_high)).nonzero()[0]

            left_lane_pixels_id.append(good_left_pixel_id)
            right_lane_pixels_id.append(good_right_pixel_id)

            if len(good_left_pixel_id) >= self.min_pixel_recenter_:
                x_left_center = int(np.mean(non_zero_pixels_x[good_left_pixel_id]))
            if len(good_right_pixel_id) >= self.min_pixel_recenter_:
                x_right_center = int(np.mean(non_zero_pixels_x[good_right_pixel_id]))

        left_lane_pixels_id = np.concatenate(left_lane_pixels_id)
        right_lane_pixels_id = np.concatenate(right_lane_pixels_id)

        x_left_lane_pixel = non_zero_pixels_x[left_lane_pixels_id]
        y_left_lane_pixel = non_zero_pixels_y[left_lane_pixels_id]
        x_right_lane_pixel = non_zero_pixels_x[right_lane_pixels_id]
        y_right_lane_pixel = non_zero_pixels_y[right_lane_pixels_id]

        self.left_lane_ = np.polyfit(y_left_lane_pixel, x_left_lane_pixel, 2)
        self.right_lane_ = np.polyfit(y_right_lane_pixel, x_right_lane_pixel, 2)

        if plot:
            cv.imshow('Image with Sliding Window', frame_sliding_window)
            cv.waitKey(0)

    def get_lane_line(self, frame, plot=False):
        non_zero_pixels = np.nonzero(frame)
        non_zero_pixels_y = non_zero_pixels[0]
        non_zero_pixels_x = non_zero_pixels[1]

        y_list = np.linspace(0, self.height_ - 1, self.height_)
        x_left_list = self.left_lane_[0] * non_zero_pixels_y ** 2 + self.left_lane_[1] * \
                      non_zero_pixels_y + self.left_lane_[2]
        x_right_list = self.right_lane_[0] * non_zero_pixels_y ** 2 + self.right_lane_[1] * \
                       non_zero_pixels_y + self.right_lane_[2]

        left_lane_pixels_id = ((non_zero_pixels_x < (x_left_list + self.margin_)) &
                               (non_zero_pixels_x > (x_left_list - self.margin_)))
        right_lane_pixels_id = ((non_zero_pixels_x < (x_right_list + self.margin_)) &
                                (non_zero_pixels_x > (x_right_list - self.margin_)))

        x_left_lane_pixel = non_zero_pixels_x[left_lane_pixels_id]
        y_left_lane_pixel = non_zero_pixels_y[left_lane_pixels_id]
        x_right_lane_pixel = non_zero_pixels_x[right_lane_pixels_id]
        y_right_lane_pixel = non_zero_pixels_y[right_lane_pixels_id]

        self.left_lane_ = np.polyfit(y_left_lane_pixel, x_left_lane_pixel, 2)
        self.right_lane_ = np.polyfit(y_right_lane_pixel, x_right_lane_pixel, 2)

        if plot:
            x_left_list = self.left_lane_[0] * y_list ** 2 + self.left_lane_[1] * y_list + \
                          self.left_lane_[2]
            x_right_list = self.right_lane_[0] * y_list ** 2 + self.right_lane_[1] * y_list + \
                           self.right_lane_[2]

            out_img = np.dstack((frame, frame, frame)) * 255
            window_img = np.zeros_like(out_img)

            # Add color to the left and right line pixels
            out_img[non_zero_pixels_y[left_lane_pixels_id],
                    non_zero_pixels_x[left_lane_pixels_id]] = [255, 0, 0]
            out_img[non_zero_pixels_y[right_lane_pixels_id],
                    non_zero_pixels_x[right_lane_pixels_id]] = [0, 0, 255]
            # Create a polygon to show the search window area, and recast
            # the x and y points into a usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([
                x_left_list - self.margin_, y_list]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                x_left_list + self.margin_, y_list])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([
                x_right_list - self.margin_, y_list]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                x_right_list + self.margin_, y_list])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the figures
            plt.figure()
            plt.imshow(result)
            plt.plot(x_left_list, y_list, color='yellow')
            plt.plot(x_right_list, y_list, color='yellow')
            plt.show()

    def detect_img(self, frame):
        dim = frame.shape
        self.height_ = dim[0]
        self.width_ = dim[1]
        self.min_pixel_recenter_ = int(1 / 48 * self.width_)
        self.margin_ = int(1 / 12 * self.width_)

        hls = cv.cvtColor(frame, cv.COLOR_RGB2HLS)
        thresh_img = util.thresh_edge(hls, frame)

        self.roi_select_img_ = thresh_img.copy()
        self.set_roi(self.roi_select_img_)

        warped_img = util.transform_perspective(thresh_img, self.roi_, self.roi_transform_,
                                                plot=False)

        self.detect_lane_pixels(warped_img, plot=False)
        self.get_lane_line(warped_img, plot=True)

        return thresh_img

    def detect(self, path):
        self.lane_file_ = []
        self.roi_select_img_ = []
        self.roi_ = []
        self.roi_transform_ = []
        self.load_lane(path)
        self.left_lane_ = []
        self.right_lane_ = []
        if self.mode_ == 'image':
            return self.detect_img(self.lane_file_)


lane_detector = Lane()
lane_detector.set_mode('image')
result = lane_detector.detect('./original_lane_detection_5.jpg')

# cv.imshow("Image", result)
# cv.waitKey(0)
# print(lane_detector.roi_)
# print(lane_detector.roi_transform_)
cv.destroyAllWindows()
