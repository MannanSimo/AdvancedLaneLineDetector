"""
Provide camera calibration and exection of AdvancedLaneLineDetector instance.
"""

import glob
import os.path as path
import pickle

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from advanced_lane_line_detector import AdvancedLaneLineDetector


TIME_WINDOW = 10  # results are averaged over this number of frames


def lazy_calibration(func):
    """Cache facility for calibration activities."""
    calibration_cache = '../camera_calibration_data/calibration_data.pickle'

    def wrapper(*args, **kwargs):
        if path.exists(calibration_cache):
            print('Loading cached camera calibration...', end=' ')
            with open(calibration_cache, 'rb') as dump_file:
                calibration = pickle.load(dump_file)
        else:
            print('Computing camera calibration...', end=' ')
            calibration = func(*args, **kwargs)
            with open(calibration_cache, 'wb') as dump_file:
                pickle.dump(calibration, dump_file)
        print('Done.')
        return calibration

    return wrapper


@lazy_calibration
def calibrate_camera(calib_images_dir: str):
    """
    Calibrate the camera given a directory containing calibration chessboards.
    """
    err_msg = F'"{calib_images_dir}" must exist and contain calibration images.'

    assert path.exists(calib_images_dir), err_msg

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(path.join(calib_images_dir, 'calibration*.jpg'))

    for filename in images:

        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pattern_found, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if pattern_found is True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )


def main():
    """Encapsulate the main workflow of the script."""
    selector = 'project'
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        calib_images_dir='../camera_calibration_data'
    )

    line_detector = AdvancedLaneLineDetector(
        TIME_WINDOW,
        ret,
        mtx,
        dist,
        rvecs,
        tvecs
    )

    clip = VideoFileClip(
        '../{}_video.mp4'.format(selector)).fl_image(line_detector.process)

    clip.write_videofile(
        '../output_{}.mp4'.format(selector),
        audio=False
    )


if __name__ == '__main__':
    main()
