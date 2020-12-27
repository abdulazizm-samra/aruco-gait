from datetime import datetime

import cv2
import numpy as np
from cv2 import aruco as ar

# STREAM = 'capture.mp4'
STREAM = 'maher/rod.mp4'
MARKER_ID = 13
MARKER_LENGTH = 90.0
CAMERA_COFF_FILE = 'camera_coff20200717173758974923.npz'

cap = cv2.VideoCapture(STREAM)

coff_file = np.load(CAMERA_COFF_FILE)
mtx = coff_file['mtx']
dist = coff_file['dist']


def nothing(x):
    pass


cv2.namedWindow('fig', cv2.WINDOW_FREERATIO)
cv2.createTrackbar('x', 'fig', 200, 400, nothing)
cv2.createTrackbar('y', 'fig', 0, 600, nothing)
cv2.createTrackbar('z', 'fig', 200, 400, nothing)

aruco_dict = ar.getPredefinedDictionary(ar.DICT_4X4_50)
aruco_parameters = ar.DetectorParameters_create()

if __name__ == '__main__':
    cv2.startWindowThread()
    while True:
        x = cv2.getTrackbarPos('x','fig') - 200
        y = - cv2.getTrackbarPos('y','fig')
        z = cv2.getTrackbarPos('z','fig') - 200
        ret, img = cap.read()
        #img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        if ret:
            corners, ids, rejected = ar.detectMarkers(img, aruco_dict, parameters=aruco_parameters)
            ar.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
            if ids is not None :
                ids = list(ids)
                if MARKER_ID in ids:
                    idx = ids.index(MARKER_ID)
                    r, t, _ = ar.estimatePoseSingleMarkers(corners[idx],
                                                           MARKER_LENGTH,
                                                           mtx, dist)
                    R, J = cv2.Rodrigues(r)
                    p = cv2.projectPoints(np.array([[0,0,0],
                                                    [x,y,z]],dtype=np.float32),
                                          R, t,
                                          mtx, dist)
                    p = p[0].reshape(2,-1).astype(np.int)
                    cv2.drawMarker(img,tuple(p[0]),(0,0,255))
                    cv2.drawMarker(img,tuple(p[1]),(255,0,255),thickness=2,markerType=cv2.MARKER_STAR)
            cv2.imshow('fig',img)
            cmd = cv2.waitKey(20)
            if cmd & 0xFF == ord('k'):
                timestamp = str(datetime.now())
                for p in ':- .':
                    timestamp = timestamp.replace(p, '')

                np.savez(f'rod_coff{timestamp}.npz',
                         marker_id = MARKER_ID,
                         tvec = np.array([x,y,z],dtype=np.float32))
        else:
            print('Cannot read image ...')
            cv2.destroyAllWindows()
            exit()
