import cv2 as cv
import numpy as np
from util import util_cv as util
from matplotlib import pyplot as plt
import os

# the following values should be set according to the videoing condition
# the parameters for our testing videos are provided
##### for testing video 1
# Y_METER_PER_PIXEL = 0.03
# X_METER_PER_PIXEL = 0.002
##### for testing video 2
Y_METER_PER_PIXEL = 0.025
X_METER_PER_PIXEL = 0.001


class LaneDetectorCV:
    def __init__(self):
        self.mode_ = 'image'
        self.lane_file_ = []
        self.roi_select_img_ = []
        self.roi_ = []
        self.roi_transform_ = []
        self.window_num_ = 10
        self.height_ = -1
        self.width_ = -1
        self.min_pixel_recenter_ = -1
        self.margin_ = -1
        self.left_lane_ = []
        self.right_lane_ = []
        self.left_curve_ = -1
        self.right_curve_ = -1
        self.center_offset_ = -1

    def set_mode(self, mode):
        """
        set the mode of the lane detector

        :param mode: new mode to be set
        """
        self.mode_ = mode

    def load_lane(self, file_path):
        """
        set the image or video to member variable

        :param file_path: input image/video path
        """
        if self.mode_ == 'image':
            try:
                self.lane_file_ = cv.imread(file_path)
            except FileNotFoundError:
                print("No file or directory with the name {}".format(file_path))
        else:
            self.lane_file_ = cv.VideoCapture(file_path)
            assert self.lane_file_.isOpened(), 'Cannot capture source'

    def click_event(self, event, x, y, flags, params):
        """
        display images and store the pixel coordinate user click by the left mouse

        :param event: event happening
        :param x: x coordinate
        :param y: y coordinate
        :param flags: flags
        :param params: params
        """
        # checking for left mouse clicks
        if event == cv.EVENT_LBUTTONDOWN:
            # displaying the coordinates on the Shell
            self.roi_.append([x, y])
            print(x, ' ', y)

            self.roi_select_img_ = cv.circle(self.roi_select_img_, (x, y), radius=5,
                                             color=(255, 0, 0),
                                             thickness=-1)
            cv.imshow('image', self.roi_select_img_)

    def manual_set_roi(self, array):
        """
        manually set pre-defined region-of-interest coordinates

        :param array: user-provided array for region-of-interest coordinate
        """
        self.roi_ = np.float32(array)

    def set_roi(self, frame):
        """
        set region-of-interest coordinates using left mouse click on the frame

        :param frame: user provided frame
        """
        cv.imshow('image', frame)
        cv.setMouseCallback('image', self.click_event)
        cv.waitKey(0)

        self.roi_ = np.float32(self.roi_)

    def detect_lane_pixels(self, frame, plot=False):
        """
        obtain best-fit functions for left and right lanes using sliding window

        :param frame: user provided frame
        :param plot: flag indicating whether a plot of the result is needed. Default value is False
        """
        window_height = np.floor(self.height_ / self.window_num_)

        # get x and y coordinates of nonzero pixels in the frame
        non_zero_pixels = np.nonzero(frame)
        non_zero_pixels_y = non_zero_pixels[0]
        non_zero_pixels_x = non_zero_pixels[1]

        left_lane_pixels_id = []
        right_lane_pixels_id = []
        frame_sliding_window = frame.copy()

        x_left_center, x_right_center = util.calculate_histogram_peak(frame)  # get two peaks

        for window in range(self.window_num_):  # iterate over all windows
            # get the bounding coordinates of the left and right windows on the current window level
            window_y_low = self.height_ - window * window_height
            window_y_high = self.height_ - (window + 1) * window_height
            window_x_left_low = x_left_center - self.margin_
            window_x_left_high = x_left_center + self.margin_
            window_x_right_low = x_right_center - self.margin_
            window_x_right_high = x_right_center + self.margin_

            if plot:  # add window to the frame if flag plot is TRUE
                cv.rectangle(frame_sliding_window, (int(window_x_left_low), int(window_y_low)), (
                    int(window_x_left_high), int(window_y_high)), (255, 255, 255), 2)
                cv.rectangle(frame_sliding_window, (int(window_x_right_low), int(window_y_low)), (
                    int(window_x_right_high), int(window_y_high)), (255, 255, 255), 2)

            # filter valid lane pixels
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

            # update the window center if the lane pixel exceeds a pre-defined number
            if len(good_left_pixel_id) >= self.min_pixel_recenter_:
                x_left_center = int(np.mean(non_zero_pixels_x[good_left_pixel_id]))
            if len(good_right_pixel_id) >= self.min_pixel_recenter_:
                x_right_center = int(np.mean(non_zero_pixels_x[good_right_pixel_id]))

        # convert list of lists to one-dimensional numpy array
        left_lane_pixels_id = np.concatenate(left_lane_pixels_id)
        right_lane_pixels_id = np.concatenate(right_lane_pixels_id)
        # get x and y coordinates of left and right lanes
        x_left_lane_pixel = non_zero_pixels_x[left_lane_pixels_id]
        y_left_lane_pixel = non_zero_pixels_y[left_lane_pixels_id]
        x_right_lane_pixel = non_zero_pixels_x[right_lane_pixels_id]
        y_right_lane_pixel = non_zero_pixels_y[right_lane_pixels_id]

        # find best-fitting functions for both lanes
        self.left_lane_ = np.polyfit(y_left_lane_pixel, x_left_lane_pixel, 2)
        self.right_lane_ = np.polyfit(y_right_lane_pixel, x_right_lane_pixel, 2)

        if plot:
            cv.imshow('Image with Sliding Window', frame_sliding_window)
            cv.waitKey(0)

    def get_lane_line(self, frame, plot=False):
        """
        obtain more precise best-fit functions for left and right lanes and calculate curvature and
        center offset information

        :param frame: user provided frame
        :param plot: flag indicating whether a plot of the result is needed. Default value is False
        """
        non_zero_pixels = np.nonzero(frame)
        non_zero_pixels_y = non_zero_pixels[0]
        non_zero_pixels_x = non_zero_pixels[1]

        y_list = np.linspace(0, self.height_ - 1, self.height_)
        x_left_list = self.left_lane_[0] * non_zero_pixels_y ** 2 + self.left_lane_[1] * \
                      non_zero_pixels_y + self.left_lane_[2]
        x_right_list = self.right_lane_[0] * non_zero_pixels_y ** 2 + self.right_lane_[1] * \
                       non_zero_pixels_y + self.right_lane_[2]

        # filter out valid lane pixels
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

        # calculate curvature information
        self.left_curve_, self.right_curve_ = util.calculate_curvature(self.height_,
                                                                       x_left_lane_pixel,
                                                                       x_right_lane_pixel,
                                                                       y_left_lane_pixel,
                                                                       y_right_lane_pixel,
                                                                       Y_METER_PER_PIXEL,
                                                                       X_METER_PER_PIXEL)
        # calculate center offset information
        self.center_offset_ = util.calculate_center_offset(self.height_, self.width_,
                                                           self.roi_transform_, self.roi_,
                                                           self.left_lane_, self.right_lane_,
                                                           X_METER_PER_PIXEL)

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
            left_line_left_bound = np.array([np.transpose(np.vstack([
                x_left_list - self.margin_, y_list]))])
            left_line_right_bound = np.array([np.flipud(np.transpose(np.vstack([
                x_left_list + self.margin_, y_list])))])
            left_line_pts = np.hstack((left_line_left_bound, left_line_right_bound))
            right_line_left_bound = np.array([np.transpose(np.vstack([
                x_right_list - self.margin_, y_list]))])
            right_line_right_bound = np.array([np.flipud(np.transpose(np.vstack([
                x_right_list + self.margin_, y_list])))])
            right_line_pts = np.hstack((right_line_left_bound, right_line_right_bound))

            # Draw the lane onto the warped blank image
            cv.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            lane_line_img = cv.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the figures
            plt.figure()
            plt.imshow(lane_line_img)
            plt.plot(x_left_list, y_list, color='yellow')
            plt.plot(x_right_list, y_list, color='yellow')
            plt.show()

    def overlay_lane(self, frame, plot=False):
        """
        overlay the lane segmentation, together curvature and center offset, on the original frame

        :param frame: user provided frame
        :param plot: flag indicating whether a plot of the result is needed. Default value is False
        :return: original frame with lane segmentation
        """
        warp_zero = np.zeros_like(frame[:, :, 0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        y_list = np.linspace(0, self.height_ - 1, self.height_)
        x_left_list = self.left_lane_[0] * y_list ** 2 + self.left_lane_[1] * y_list + \
                      self.left_lane_[2]
        x_right_list = self.right_lane_[0] * y_list ** 2 + self.right_lane_[1] * y_list + \
                       self.right_lane_[2]
        # get bounds for left and right lanes
        left_line_bound = np.array([np.transpose(np.vstack([x_left_list, y_list]))])
        right_line_bound = np.array([np.flipud(np.transpose(np.vstack([x_right_list, y_list])))])
        lane_pts = np.hstack((left_line_bound, right_line_bound))
        # fill in-between left and right lanes
        cv.fillPoly(color_warp, np.int_([lane_pts]), (0, 255, 0))

        # transform the lane-information image to the perspective of the original frame
        warped_img = util.transform_perspective(color_warp, self.roi_transform_, self.roi_)
        _, warped_img = cv.threshold(warped_img, 127, 255, cv.THRESH_BINARY)
        # combine the lane information image with the original image using certain weight
        lane_img = cv.addWeighted(frame, 1, warped_img, 0.3, 0)
        cv.putText(lane_img, 'Curve Radius: ' + str((self.left_curve_ + self.right_curve_)
                                                    / 2)[:7] + ' m',
                   (int((5 / 600) * self.width_), int((20 / 338) * self.height_)),
                   cv.FONT_HERSHEY_SIMPLEX, (float((0.5 / 600) * self.width_)),
                   (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(lane_img, 'Center Offset: ' + str(self.center_offset_)[:7] + ' cm',
                   (int((5 / 600) * self.width_), int((40 / 338) * self.height_)),
                   cv.FONT_HERSHEY_SIMPLEX, (float((0.5 / 600) * self.width_)),
                   (255, 255, 255), 2, cv.LINE_AA)

        if plot:
            cv.imshow('image', lane_img)
            cv.waitKey(0)

        return lane_img

    def detect_img(self, src, frame, roi=True, output_path=None, plot=True, manual=False,
                   array=None):
        """
        process image and output with lane segmentation

        :param src: the image path
        :param frame: the user provided frame
        :param roi: flag indicating whether to region of interest selection is needed.
                    Default is True
        :param output_path: path for the image to be output. None indicating no output is needed.
                            Default is None
        :param plot: flag indicating whether a plot of the result is needed. Default is True
        :param manual: flag indicating whether the region-of-interest coordinates will be set by
                       pre-defined arrays. Default is False
        :param array: pre-defined region-of-interest coordinates
        :return: original frame with lane segmentation
        """
        dim = frame.shape
        self.height_ = dim[0]
        self.width_ = dim[1]
        self.min_pixel_recenter_ = int(1 / 48 * self.width_)
        self.margin_ = int(1 / 12 * self.width_)

        if roi:  # check if region-of-interest selection is needed
            self.roi_transform_.append([0.2 * self.width_, 0])
            self.roi_transform_.append([0.2 * self.width_, self.height_])
            self.roi_transform_.append([0.8 * self.width_, self.height_])
            self.roi_transform_.append([0.8 * self.width_, 0])
            self.roi_transform_ = np.float32(self.roi_transform_)

            self.roi_select_img_ = frame.copy()
            if manual and array is not None:  # if flag manual is TRUE
                self.manual_set_roi(array)  # set pre-defined region-of-interest coordinates
            else:
                self.set_roi(self.roi_select_img_)  # select region-of-interest using mouse

        # transform perspective
        warped_img = util.transform_perspective(frame, self.roi_, self.roi_transform_)
        # convert to HLS representation
        hls = cv.cvtColor(warped_img, cv.COLOR_RGB2HLS)
        warped_img = util.thresh_edge(hls, warped_img)

        _, warped_img = cv.threshold(warped_img, 127, 255, cv.THRESH_BINARY)

        self.detect_lane_pixels(warped_img, plot=False)
        self.get_lane_line(warped_img, plot=False)

        lane_img = self.overlay_lane(frame, plot=plot)

        if output_path is not None:  # output is needed
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_img = '{}/cv_det_{}'.format(output_path, src.split('/')[-1])
            cv.imwrite(output_img, lane_img)

        return lane_img

    def detect_vid(self, src, cap, output_path=None, manual=False, array=None):
        """
        process video and obtain lane segmentation of each frame

        :param src: the video path
        :param cap: the caption object containing the video
        :param output_path: path for the image to be output. None indicating no output is needed.
                            Default is None
        :param manual: flag indicating whether the region-of-interest coordinates will be set by
                       pre-defined arrays. Default is False
        :param array: pre-defined region-of-interest coordinates
        """
        # Set up output video
        # make the destination directory if not exist already
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc(*'MJPG')  # Define the codec for output video
            output_vid = '{}/cv_det_{}.avi'.format(output_path, src.split('/')[-1].split('.')[0])
            out = cv.VideoWriter(output_vid, fourcc, 20.0, (cols, rows))

        frame_cnt = 0
        while cap.isOpened():  # iterate over all video frames
            ret, frame = cap.read()
            if ret:
                if frame_cnt == 0:  # set region-of-interest coordinates only once
                    if manual and array is not None:
                        lane_img = self.detect_img(src, frame, roi=True, plot=False, manual=True,
                                                   array=array)
                    else:
                        lane_img = self.detect_img(src, frame, roi=True, plot=False)
                else:
                    lane_img = self.detect_img(src, frame, roi=False, plot=False)

                frame_cnt += 1
                if output_path is not None:
                    out.write(lane_img)
                cv.imshow("lane video", lane_img)

                key = cv.waitKey(1)  # allow early termination
                if key == 27:
                    if output_path is not None:
                        out.release()
                    cap.release()
                    cv.destroyAllWindows()
                    break
            else:
                break

        cap.release()
        if output_path is not None:
            out.release()

    def detect(self, path, output_path=None, manual=False, array=None):
        """
        perform lane detection based on the input type (image/video)

        :param path: the image/video path
        :param output_path: path for the image to be output. None indicating no output is needed.
                            Default is None
        :param manual: flag indicating whether the region-of-interest coordinates will be set by
                       pre-defined arrays. Default is False
        :param array: pre-defined region-of-interest coordinates
        """
        self.lane_file_ = []
        self.roi_select_img_ = []
        self.roi_ = []
        self.roi_transform_ = []
        self.load_lane(path)
        self.left_lane_ = []
        self.right_lane_ = []
        if self.mode_ == 'image':
            self.detect_img(path, self.lane_file_, output_path=output_path)
        else:
            self.detect_vid(path, self.lane_file_, output_path=output_path, manual=manual,
                            array=array)
        cv.destroyAllWindows()
