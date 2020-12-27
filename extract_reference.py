import cv2
import cv2.aruco as ar
import numpy as np

ref_id = 5
cap = cv2.VideoCapture('maher4/reference.mp4')
parameters = ar.DetectorParameters_create()
aruco_dict = ar.getPredefinedDictionary(ar.DICT_4X4_50)
coeff_file = np.load('maher4\camera_coff20200906090813491112.npz')
mtx = coeff_file['mtx']
dist = coeff_file['dist']

flip_axis_R = np.array([[ 0,-1, 0],
                        [ 0, 0, 1],
                        [-1, 0, 0]],
                       dtype=np.float32)


if __name__=="__main__":
    ret = True
    R_ref, t_ref = None, None
    cv2.namedWindow('fig',cv2.WINDOW_FREERATIO)
    cv2.startWindowThread()
    while ret:
        ret , img = cap.read()
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        corners, ids, rejected = ar.detectMarkers(
            img,
            parameters=parameters,
            dictionary=aruco_dict
        )
        ar.drawDetectedMarkers(img,corners,ids)
        if ids is not None:
            id_list = list (ids.reshape(-1))
            if(ref_id in id_list):
                r, t, trash = ar.estimatePoseSingleMarkers(corners[id_list.index(ref_id)],
                                             markerLength=120.0,
                                             cameraMatrix=mtx,
                                             distCoeffs=dist)
                ar.drawAxis(img,mtx,dist,r,t,60)
                print(r)
                R_ref , jacc= cv2.Rodrigues(r)
                t_ref = t.reshape(-1)
        cv2.imshow('fig',img)
        cmd = cv2.waitKey(50)
        if (cmd & 0xff)==ord('k') and R_ref is not None and t_ref is not None:
            print("#############SAVED#############")
            print(flip_axis_R @ R_ref)
            print(t_ref)
            print("###############################")
            np.savez('maher4/reference.npz', R_ref=flip_axis_R @ R_ref, t_ref=t_ref)