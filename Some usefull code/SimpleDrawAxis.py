"""for i in range(len(corners)):
    frame = aruco.drawAxis(frame, newcameramtx, dist, rvecs[i], tvecs[i], 1.5)
    cv2.putText(info_screen, "id: {:3d}, rv[{:3d}]: {:+f} {:+f} {:+f}, tv[{:3d}]: {:+f} {:+f} {:+f}".format(
        ids[i][0], i, *rvecs[0].tolist()[0], i, *tvecs[0].tolist()[0]),
                (10, 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)"""