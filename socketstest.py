import socket
import json
import cv2
from cv2 import aruco
import numpy as np


class DrawClass:
    mtx = None
    dist = None

    def __init__(self, mtx, dist, w=640, h=480):  # w=640, h=480 is kostyl'
        self.mtx, roi = cv2.getOptimalNewCameraMatrix(np.array(mtx), np.array(dist), (w, h), 1, (w, h), True)
        # self.mtx = np.array(mtx)
        self.dist = np.array(dist)

    @staticmethod
    def draw_markers(frame, corners, ids):
        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
        return frame


marker_length = 0.46  # 0.046
marker_separation = 0.006

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    exit(1)

print('cam opened')

# sock = socket.socket()
#
# sock.bind(('', 9090))
#
# sock.listen(1)
# conn, addr = sock.accept()

ardict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
board = aruco.GridBoard_create(4, 5, marker_length, marker_separation, ardict)

parameters = aruco.DetectorParameters_create()
parameters.minMarkerPerimeterRate = 0.03

# load camera parameters
with open("calib.json", 'r') as f:
    data = json.load(f)

dr = DrawClass(data.get('camera_matrix'), data.get('dist_coeff'))
# and make some variables for aruco
mtx = dr.mtx
dist = dr.dist

w = 640
h = 480
mtx, roi = cv2.getOptimalNewCameraMatrix(np.array(mtx), np.array(dist), (w, h), 1, (w, h), True)

tv3 = np.zeros_like(3)

n = 13

queue = np.zeros((n, 3))

while cam.isOpened():
    k = cv2.waitKey(16)

    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    corners, ids, rejected_img_points = aruco.detectMarkers(gray, ardict, parameters=parameters)

    if corners is not None:
        frame = dr.draw_markers(frame, corners, ids)

        r_vecs, t_vecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

        if t_vecs is not None:
            try:
                frame = aruco.drawAxis(frame, mtx, dist, r_vecs, t_vecs, 5)
            except Exception as E:
                print(t_vecs, r_vecs, E)

            if k == ord('c'):
                tv3 = t_vecs[0][0]

            rot_m = cv2.Rodrigues(r_vecs[0][0])[0]

            offset = np.array([marker_length / 2, marker_length / 2, 0])

            tv_from_m = np.dot(rot_m, offset.T).T

            tv = tv_from_m + t_vecs[0]

            # print(tv)

            aruco.drawAxis(frame, mtx, dist, r_vecs[0][0], tv[0], 5)

            queue[0:n - 1] = queue[1:]
            queue[n - 1] = t_vecs[0][0] - tv3

            send_data = np.sum(queue, axis=0) / n
            send_data[0] *= 10
            send_data[1] *= 10
            send_data[2] *= -1
            send_data = np.array(send_data * 100, dtype=np.int) / 100

            # conn.send((" ".join(map(str, t_vecs[0][0] - tv3)) + '\n').encode())
            # conn.send((" ".join(map(str, send_data)) + '\n').encode())

            print('sent:', " ".join(map(str, t_vecs[0][0] - tv3)))
            # print('tv3:', tv3)

    cv2.line(frame, (320, 240), (320, 240), (0, 0, 0), 4)
    cv2.imshow("frame", frame)

    if k == ord('q'):
        # conn.close()
        cam.release()
        break


