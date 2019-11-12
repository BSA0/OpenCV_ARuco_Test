import cv2
from cv2 import aruco

import numpy as np
import json

cam = cv2.VideoCapture(0)

ardict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
parameters = aruco.DetectorParameters_create()

counter = []
corners_list = []
id_list = []
first = True
shape = None

save_images = True
use_images = False
i = 0

if use_images:
    cam.release()

while cam.isOpened() or use_images:
    if not use_images:
        ret, frame = cam.read()
    else:
        frame = cv2.imread(str(i) + '.bmp')
        i += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if save_images and not use_images:
        try:
            cv2.imwrite(str(i) + '.bmp', frame)
            i += 1
        except Exception as e:
            break

    shape = gray.shape

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ardict, parameters=parameters)

    if corners is not None and ids is not None:
        if first:
            first = False
            corners_list = corners
            id_list = ids
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        counter.append(len(ids))
        if save_images and not use_images:
            cv2.imwrite(str(i) + '.bmp', frame)
            i += 1

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    # print(len(corners_list))
    cv2.imshow("cam", frame_markers)

    k = cv2.waitKey(17)
    # print(k)
    if k == 113:
        cam.release()
        break

cv2.destroyAllWindows()

print("Found {} unique markers ({})".format(len(np.unique(id_list)), np.unique(id_list)))

print("Calibration started on {} frames, please, wait".format(len(counter)))
board = aruco.GridBoard_create(4, 5, 4.5, 0.6, ardict)
counter = np.array(counter)
try:
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, shape, None, None)

    print("Camera matrix is \n", mtx,
          "\n And is stored in calibration.json file along with distortion coefficients : \n",
          dist)

    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

    with open("calibration.json", "w") as f:
        json.dump(data, f)
except Exception as e:
    print("Calibration failed, reason: {}".format(e))
