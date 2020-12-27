from datetime import date, time, datetime
import urllib.request
print('ready')
import cv2
import numpy as np
import pickle

def get_image(url):
    return cv2.imdecode(
        np.array(
            bytearray(
                urllib.request.urlopen(url).read()
            ),dtype=np.int8
        ),-1
    )

print('ready')
#cap = cv2.VideoCapture('http://192.168.1.6:8080/video')
cap = cv2.VideoCapture('maher4\camera_calib.mp4')

img_points = []
obj_points = []

x = 5
y = 8
d = 30 # mm

objp = np.zeros((x * y, 3), np.float32)
objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2) * d

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

image_count = 0

if __name__=="__main__":
    cv2.namedWindow('fig',cv2.WINDOW_FREERATIO)
    cv2.startWindowThread()
    print('ready')
    while True:
        ret , img = cap.read()
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        if ret:
            #img = get_image('http://192.168.1.7:8080/shot.jpg')
            cv2.imshow('fig',img)
            cmd = (cv2.waitKey(1) & 0xff)
        else: cmd = 'q'
        if cmd == ord('t'):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret , corners = cv2.findChessboardCorners(gray,(x,y))
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img = cv2.drawChessboardCorners(img, (x, y), corners2, ret)
                cv2.imshow('fig',img)
                print("add(y/n) ? ")
                cmd2 = (cv2.waitKey(10000) & 0xff)
                if cmd2 ==ord("y"):
                    obj_points.append(objp)
                    img_points.append(corners2)
                    image_count = image_count + 1
                    print(f"Number of images{image_count}")
                    ret = False
            else:
                print("can't detect chessboard !!")
        elif cmd == ord('c'):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret,  mtx,  dist,  rvecs,  tvecs = \
                cv2.calibrateCamera( obj_points,  img_points,  gray.shape[::-1], None, None)
            print(ret,  mtx,  dist,  rvecs,  tvecs, sep='\n\n')
            timestamp = str(datetime.now())
            for p in ':- .':
                timestamp = timestamp.replace(p,'')

            np.savez(f'camera_coff{timestamp}.npz', mtx = mtx, dist = dist)

        elif cmd == ord('q'):
            cv2.destroyAllWindows()
            exit()
