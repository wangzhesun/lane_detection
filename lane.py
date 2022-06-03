import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import util


class Lane:
    def __init__(self):
        self.path_ = ''
        self.mode_ = 'image'
        self.lane_file_ = -1
        self.thresh_img_ = -1

    def set_mode(self, mode):
        self.mode_ = mode

    def load_lane(self, file_path):
        self.lane_file_ = -1
        if self.mode_ == 'image':
            try:
                self.lane_file_ = cv.imread(file_path)
            except FileNotFoundError:
                print("No file or directory with the name {}".format(file_path))
        else:
            self.lane_file_ = cv.VideoCapture(file_path)
            assert self.lane_file_.isOpened(), 'Cannot capture source'

    def thresh_edge(self, hls_img, orig_img):
        _, l_thresh_img = cv.threshold(hls_img[:, :, 1], 120, 255, cv.THRESH_BINARY)
        l_thresh_img = cv.GaussianBlur(l_thresh_img, (3, 3), 0)

        l_thresh_img = util.mag_thresh(l_thresh_img, sobel_kernel=3, thresh=(110, 255))

        _, s_thresh_img = cv.threshold(hls_img[:, :, 2], 80, 255, cv.THRESH_BINARY)
        _, r_thresh_img = cv.threshold(orig_img[:, :, 2], 120, 255, cv.THRESH_BINARY)

        sr_thresh_img = cv.bitwise_and(s_thresh_img, r_thresh_img)
        srl_thresh_img = cv.bitwise_or(sr_thresh_img, l_thresh_img.astype(np.uint8))

        return srl_thresh_img

    def detect_img(self):
        hls = cv.cvtColor(self.lane_file_, cv.COLOR_RGB2HLS)
        self.thresh_img_ = self.thresh_edge(hls, self.lane_file_)
        return self.thresh_img_

    def detect(self):
        if self.mode_ == 'image':
            return self.detect_img()


lane_detector = Lane()
lane_detector.set_mode('image')
lane_detector.load_lane('./original_lane_detection_5.jpg')
result = lane_detector.detect()

cv.imshow("Image", result)
cv.waitKey(0)
cv.destroyAllWindows()