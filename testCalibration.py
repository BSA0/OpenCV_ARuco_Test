import json
import cv2
from cv2 import aruco
import numpy as np
np.set_printoptions(linewidth=200)


class Marker:
    marker_id = None
    r_vec = None
    t_vec = None
    marker_corners = None

    def __init__(self, marker_id, r_vec, t_vec):
        self.marker_id = marker_id
        self.t_vec = t_vec
        self.r_vec = r_vec


class CoordinatesSystem:
    fcs_r_vec = None
    fcs_t_vec = None
    trans_matrix = None
    offset = np.zeros(3)

    def reset(self, r_vec, t_vec):
        self.fcs_r_vec = r_vec
        self.fcs_t_vec = t_vec
        self.trans_matrix = self.get_transitional_matrix()
        self.offset = -self.get_coordinates_in_cs(t_vec)

    def get_rotation_matrix(self):
        return cv2.Rodrigues(self.fcs_r_vec)[0]

    def get_transitional_matrix(self):
        trans_matrix = np.eye(4)
        trans_matrix[0:3, 0:3] = self.get_rotation_matrix().T
        trans_matrix[0:3, 3] = -self.fcs_t_vec
        trans_matrix[3, 3] = 1
        return trans_matrix

    def get_rotations(self):
        return cv2.RQDecomp3x3(self.get_rotation_matrix())[0]

    def get_coordinates_in_cs(self, t_vec):
        camera_coords = np.ones((4, 1))
        camera_coords[0:3, 0] = t_vec
        camera_coords[3, 0] = 1

        return self.trans_matrix.dot(camera_coords)[0:3].T[0] + self.offset


class DrawClass:
    mtx = None
    dist = None

    def __init__(self, mtx, dist, w=640, h=480):  # w=640, h=480 is kostyl'
        self.mtx, roi = cv2.getOptimalNewCameraMatrix(np.array(mtx), np.array(dist), (w, h), 1, (w, h), True)
        # self.mtx = np.array(mtx)
        self.dist = np.array(dist)

    # marker draw functions
    @staticmethod
    def draw_markers(frame, corners, ids):
        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
        return frame

    def draw_axis_of_markers(self, frame, markers: list, length=10):
        for marker in markers:
            frame = aruco.drawAxis(frame, self.mtx, self.dist, marker.r_vec, marker.t_vec, length)
        return frame

    def draw_axis_points(self, frame, r, t):
        img_pts, _ = cv2.projectPoints(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float),
                                       r, t, self.mtx, self.dist)
        for i in range(4):
            rgb = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
            cv2.line(frame, (int(img_pts[0][0][0]), int(img_pts[0][0][1])),
                            (int(img_pts[i][0][0]), int(img_pts[i][0][1])), rgb[i], 4)
        return frame

    @staticmethod
    def print_text(info_screen, markers):
        for marker in markers:
            cv2.putText(info_screen, "id: {:3d}, rv: {:+f} {:+f} {:+f}, tv: {:+f} {:+f} {:+f}".format(
                marker.marker_id, *marker.r_vec.tolist()[0], *marker.t_vec.tolist()[0]),
                        (10, 18 * (marker.marker_id + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return info_screen

    @staticmethod
    def draw_debug_lines(info_screen, t_vec):
        cv2.line(info_screen, (900, 200), (900, 200 + int(t_vec[0] * 10)), (0, 255, 0), 4)
        cv2.line(info_screen, (900, 200), (900 + int(t_vec[1] * 10), 200), (0, 0, 255), 4)
        cv2.line(info_screen, (900, 200), (900 + int(t_vec[1] * 10), 200 - int(t_vec[1] * 10)), (255, 0, 0), 4)
        return info_screen


def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape(
        (3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    orgRvec, orgTvec = inversePerspective(invRvec, invTvec)
    # print("rvec: ", rvec2, "tvec: ", tvec2, "\n and \n", orgRvec, orgTvec)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def main(marker_length, marker_separation):

    # opencv/aruco settings
    cam = cv2.VideoCapture(0)
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

    # print(cv2.calibrationMatrixValues(mtx, (640, 480), 3.6, 2.7))

    # globals variables
    cs = CoordinatesSystem()

    while cam.isOpened():
        k = cv2.waitKey(16)

        ret, frame = cam.read()
        info_screen = np.ones((500, 1000, 3), np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, rejected_img_points = aruco.detectMarkers(gray, ardict, parameters=parameters)

        if corners is not None:
            frame = dr.draw_markers(frame, corners, ids)

            r_vecs, t_vecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

            markers = [Marker(ids[i][0], r_vecs[i], t_vecs[i]) for i in range(len(corners))]
            frame = dr.draw_axis_of_markers(frame, markers, 1)
            info_screen = dr.print_text(info_screen, markers)

            if len(markers) == 2:
                rv, tv = relativePosition(markers[0].r_vec, markers[0].t_vec, markers[0].r_vec, markers[0].r_vec)
                print(tv)

            if len(markers) == 1 and k == ord('p'):
                cs.reset(markers[0].r_vec, markers[0].t_vec)
                print(cs.trans_matrix)

            if len(markers) == 1 and cs.fcs_r_vec is not None:
                tv3 = cs.get_coordinates_in_cs(markers[0].t_vec)
                frame = dr.draw_axis_points(frame, cs.fcs_r_vec, cs.fcs_t_vec)
                info_screen = dr.draw_debug_lines(info_screen, tv3)
                print(tv3)

        cv2.line(frame, (320, 240), (320, 240), (0, 0, 0), 4)

        cv2.imshow("frame", frame)
        cv2.imshow("info", info_screen)

        if k == ord('q'):
            cam.release()
            break


# if rvecsB is not None:
#     r, _ = cv2.Rodrigues(rvecsB, jacobian=0)
#     ypr = cv2.RQDecomp3x3(r)[0]
#     # print(ypr, tvecsB)

# _, rvecsB, tvecsB = aruco.estimatePoseBoard(corners, ids, board, mtx, dist)

# if rvecsB is not None:
#     frame = aruco.drawAxis(frame, mtx, dist, rvecsB, tvecsB, 5)
