import cv2
from cv2 import aruco
import numpy as np
import json


def main(marker_length, marker_separation):
    cam = cv2.VideoCapture(0)

    ardict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters_create()

    # create arUco board
    board = aruco.GridBoard_create(4, 5, marker_length, marker_separation, ardict)

    shape = None
    counter, corners_list, id_list = [], [], []

    while cam.isOpened():
        _, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        shape = gray.shape
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, ardict, parameters=parameters)

        # rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker, mtx, dist)

        if corners is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("123", frame)

        k = cv2.waitKey(17)

        if k != -1:
            print(k)

        if k == ord('p'):
            if len(corners_list) == 0:
                corners_list = corners
                id_list = ids
            else:
                corners_list = np.vstack((corners, corners_list))
                id_list = np.vstack((ids, id_list))
            counter.append(len(ids))

        if k == ord('q'):
            cam.release()
            break

    counter = np.array(counter)
    print(corners_list, id_list, counter, shape, sep='\n')
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, shape, None, None)

    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

    f = open("calib.json", 'w')
    json.dump(data, f)
    f.close()
