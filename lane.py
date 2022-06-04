import cv2 as cv
import numpy as np
import util
from matplotlib import pyplot as plt
import os


Y_METER_PER_PIXEL = 10
X_METER_PER_PIXEL = 10

class Lane:
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

        self.roi_ = np.float32(self.roi_)

        self.roi_transform_.append([0.25 * self.width_, 0])
        self.roi_transform_.append([0.25 * self.width_, self.height_])
        self.roi_transform_.append([0.75 * self.width_, self.height_])
        self.roi_transform_.append([0.75 * self.width_, 0])
        self.roi_transform_ = np.float32(self.roi_transform_)

    def detect_lane_pixels(self, frame, plot=False):
        window_height = np.floor(self.height_ / self.window_num_)

        non_zero_pixels = np.nonzero(frame)
        non_zero_pixels_y = non_zero_pixels[0]
        non_zero_pixels_x = non_zero_pixels[1]

        left_lane_pixels_id = []
        right_lane_pixels_id = []
        frame_sliding_window = frame.copy()

        x_left_center, x_right_center = util.calculate_histogram_peak(frame)

        for window in range(self.window_num_):
            window_y_low = self.height_ - window * window_height
            window_y_high = self.height_ - (window + 1) * window_height
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

        self.left_curve_, self.right_curve_ = util.calculate_curvature(self.height_,
                                                                       x_left_lane_pixel,
                                                                       x_right_lane_pixel,
                                                                       y_left_lane_pixel,
                                                                       y_right_lane_pixel,
                                                                       Y_METER_PER_PIXEL,
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
        warp_zero = np.zeros_like(frame[:, :, 0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        y_list = np.linspace(0, self.height_ - 1, self.height_)
        x_left_list = self.left_lane_[0] * y_list ** 2 + self.left_lane_[1] * y_list + \
                      self.left_lane_[2]
        x_right_list = self.right_lane_[0] * y_list ** 2 + self.right_lane_[1] * y_list + \
                       self.right_lane_[2]

        left_line_bound = np.array([np.transpose(np.vstack([x_left_list, y_list]))])
        right_line_bound = np.array([np.flipud(np.transpose(np.vstack([x_right_list, y_list])))])
        lane_pts = np.hstack((left_line_bound, right_line_bound))

        cv.fillPoly(color_warp, np.int_([lane_pts]), (0, 255, 0))

        warped_img = util.transform_perspective(color_warp, self.roi_transform_, self.roi_)

        lane_img = cv.addWeighted(frame, 1, warped_img, 0.3, 0)

        if plot:

            # Plot the figures
            # plt.figure()
            # plt.imshow(cv.cvtColor(lane_img, cv.COLOR_BGR2RGB))
            # plt.text()
            cv.putText(lane_img, 'Curve Radius: ' + str((self.left_curve_ + self.right_curve_)
                                                        / 2)[:7] + ' m',
                       (int((5 / 600) * self.width_), int((20 / 338) * self.height_)),
                       cv.FONT_HERSHEY_SIMPLEX, (float((0.5 / 600) * self.width_)),
                       (255, 255, 255), 2, cv.LINE_AA)
            cv.imshow('image', lane_img)
            cv.waitKey(0)
            # plt.show()

        return lane_img

    def detect_img(self, src, frame, roi=True, output_path=None, plot=True):
        dim = frame.shape
        self.height_ = dim[0]
        self.width_ = dim[1]
        self.min_pixel_recenter_ = int(1 / 48 * self.width_)
        self.margin_ = int(1 / 12 * self.width_)

        hls = cv.cvtColor(frame, cv.COLOR_RGB2HLS)
        thresh_img = util.thresh_edge(hls, frame)

        if roi:
            self.roi_select_img_ = thresh_img.copy()
            self.set_roi(self.roi_select_img_)

        warped_img = util.transform_perspective(thresh_img, self.roi_, self.roi_transform_,
                                                plot=False)

        self.detect_lane_pixels(warped_img, plot=False)
        self.get_lane_line(warped_img, plot=False)

        lane_img = self.overlay_lane(frame, plot=plot)

        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_img = '{}/det_{}'.format(output_path, src.split('/')[-1])
            cv.imwrite(output_img, lane_img)

        return lane_img

    def detect_vid(self, src, cap, output_path=None):
        n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Get frame count
        # Get width and height of video stream
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Set up output video
        # make the destination directory if not exist already
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc(*'MJPG')  # Define the codec for output video
            output_vid = '{}/det_{}.avi'.format(output_path, src.split('/')[-1].split('.')[0])
            out = cv.VideoWriter(output_vid, fourcc, 20.0, (cols, rows))

        frame_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_cnt == 0:
                    lane_img = self.detect_img(src, frame, roi=True, output_path=None, plot=False)
                else:
                    lane_img = self.detect_img(src, frame, roi=False, output_path=None, plot=False)
                frame_cnt += 1
                if output_path is not None:
                    out.write(lane_img)
                cv.imshow("lane video", lane_img)

                key = cv.waitKey(1)
                if key == 27:
                    if output_path is not None:
                        out.release()
                    cap.release()
                    cv.destroyAllWindows()
                    break
            else:
                break

        if output_path is not None:
            out.release()

    def detect(self, path, output_path=None):
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
            self.detect_vid(path, self.lane_file_, output_path)


lane_detector = Lane()
lane_detector.set_mode('image')
result = lane_detector.detect('./images/lane_7.jpg', './det')
# lane_detector.set_mode('video')
# result = lane_detector.detect('./videos/video_1.mp4', output_path='./det')

# cv.imshow("Image", result)
# cv.waitKey(0)
# print(lane_detector.roi_)
# print(lane_detector.roi_transform_)
cv.destroyAllWindows()
