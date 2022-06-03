import cv2 as cv
import numpy as np
import util


class Lane:
    def __init__(self):
        self.mode_ = 'image'
        self.lane_file_ = []
        self.roi_select_img_ = []
        self.roi_ = []
        self.roi_transform_ = []

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
        dim = frame.shape
        height = dim[0]
        width = dim[1]

        cv.imshow('image', frame)
        cv.setMouseCallback('image', self.click_event)
        cv.waitKey(0)

        self.roi_[1][1] = height
        self.roi_[2][1] = height
        self.roi_ = np.float32(self.roi_)

        self.roi_transform_.append([0.15 * width, 0])
        self.roi_transform_.append([0.15 * width, height])
        self.roi_transform_.append([0.85 * width, height])
        self.roi_transform_.append([0.85 * width, 0])
        self.roi_transform_ = np.float32(self.roi_transform_)

    def detect_img(self, frame):
        hls = cv.cvtColor(frame, cv.COLOR_RGB2HLS)
        thresh_img = util.thresh_edge(hls, frame)

        self.roi_select_img_ = thresh_img.copy()
        self.set_roi(self.roi_select_img_)

        warped_img = util.transform_perspective(thresh_img, self.roi_, self.roi_transform_)

        histogram = np.sum(warped_img[int(warped_img.shape[0] / 2):, :], axis=0)

        return thresh_img

    def detect(self, path):
        self.lane_file_ = []
        self.roi_select_img_ = []
        self.roi_ = []
        self.roi_transform_ = []
        self.load_lane(path)
        if self.mode_ == 'image':
            return self.detect_img(self.lane_file_)


lane_detector = Lane()
lane_detector.set_mode('image')
result = lane_detector.detect('./original_lane_detection_5.jpg')

# cv.imshow("Image", result)
# cv.waitKey(0)
print(lane_detector.roi_)
print(lane_detector.roi_transform_)
cv.destroyAllWindows()
