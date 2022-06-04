from detector_cv import LaneDetectorCV

if __name__ == '__main__':
    lane_detector_cv = LaneDetectorCV()
    lane_detector_cv.set_mode('image')
    lane_detector_cv.detect('./images/lane_8.jpg')
    # lane_detector_cv.set_mode('video')
    ##### best region of interest for testing video 1
    # array_vid = [[479, 285], [4, 533], [828, 538], [565, 288]]
    ##### best region of interest for testing video 2
    # array_vid = [[565, 104], [6, 366], [1144, 718], [788, 112]]
    # result = lane_detector_cv.detect('./videos/video_2.mp4', manual=True, array=array_vid)
