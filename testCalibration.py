import cv2
from cv2 import aruco

import numpy as np
import json

cam = cv2.VideoCapture(0)

with open("calibration.json") as f:
    settings = json.load(f)

mtx = np.array(settings['camera_matrix'])
dist = np.array(settings['dist_coeff'])

ardict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.GridBoard_create(4, 5, 4.5, 0.6, ardict)
parameters = aruco.DetectorParameters_create()

new_camera_mtx = None

if cam.isOpened():
    ret, img = cam.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

while cam.isOpened():
    ret, img = cam.read()

    img_aruco = img
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = im_gray.shape[:2]
    dst = cv2.undistort(im_gray, mtx, dist, None, new_camera_mtx)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, ardict, parameters=parameters)

    if corners is not None:
        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, new_camera_mtx, dist)  # For a board
        print("Rotation ", rvec, "Translation", tvec)
        if ret != 0:
            img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
            img_aruco = aruco.drawAxis(img_aruco, new_camera_mtx, dist, rvec, tvec, 10)
            # axis length 100 can be changed according to your requirement

    cv2.imshow("World co-ordinate frame axes", img_aruco)

    k = cv2.waitKey(17)
    # print(k)
    if k == 113:
        cam.release()
        break
