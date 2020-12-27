import cv2
import numpy as np
import cv2.aruco as ar
import pickle
from drawCube import drawCube
from TRCWriter import TRCWriter

VIDEO_SOURCE = 'maher4\IK\IK.mp4'
#VIDEO_SOURCE = 'maher4/reference.mp4'
CALIBRATION_RESULT = 'maher4/result_corrected.pkl'
CAMERA_COEFF_FILE = 'maher4\camera_coff20200906090813491112.npz'
REFERENCE_FILE = 'maher4/reference.npz'
DISTINATION_FILE = 'maher4/result_30_fps_final.trc'
FPS = 30

cap = cv2.VideoCapture(VIDEO_SOURCE)
fh = open(CALIBRATION_RESULT,'rb')
cameraCoeffFile = np.load(CAMERA_COEFF_FILE)
refernceFile = np.load(REFERENCE_FILE)
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (1080,1920))

mtx = cameraCoeffFile['mtx']
dist = cameraCoeffFile['dist']
R_ref = refernceFile['R_ref']
t_ref = refernceFile['t_ref']

anatomicalMarkersFullPath = [
            'pelvis/V.SACRAL',
            'pelvis/R.ASIS',
            'pelvis/L.ASIS',
            'femur_r/R.Femur.Tubercle',
            'femur_r/R.Knee.Lat',
            'femur_r/R.Knee.Med',
            'femur_l/L.Femur.Tubercle',
            'femur_l/L.Knee.Lat',
            'femur_l/L.Knee.Med',
            'tibia_r/R.Shank.Front',
            'tibia_r/R.Shank.Upper',
            'tibia_r/R.Ankle.Lat',
            'tibia_r/R.Ankle.Med',
            'tibia_l/L.Shank.Front',
            'tibia_l/L.Shank.Upper',
            'tibia_l/L.Ankle.Lat',
            'tibia_l/L.Ankle.Med'
        ]

anatomicalMarkerSegmentDict = dict()

for s in anatomicalMarkersFullPath:
    ss = s.split('/')
    if ss[0] in anatomicalMarkerSegmentDict.keys():
        anatomicalMarkerSegmentDict[ss[0]].append(ss[1])
    else:
        anatomicalMarkerSegmentDict[ss[0]]=[ss[1]]

calibrationResult = pickle.load(fh)
markerIdDict = calibrationResult['marker_ids']
anatomicalMarkerVecDict = calibrationResult['anatomical_marker_vec']
anatomicalMarkerVecDict['V.SACRAL'] = np.array([0,0,0],dtype=np.float)

#########
# MANUAL VECTOR EDIT
########
MANUAL = False
if MANUAL:
    #anatomicalMarkerVecDict['L.Ankle.Lat'] = np.array([-20,-195,-20],dtype=np.float)
    anatomicalMarkerVecDict['L.ASIS'] *= np.array([1,-1,1],dtype=np.float)
    anatomicalMarkerVecDict['R.ASIS'] *= np.array([1,-1,1],dtype=np.float)
    anatomicalMarkerVecDict['L.Femur.Tubercle'] *= np.array([-1,1,1],dtype=np.float)
    anatomicalMarkerVecDict['L.Knee.Lat'] *= np.array([-1,1,1],dtype=np.float)
    anatomicalMarkerVecDict['L.Knee.Med'] *= np.array([-1,1,1],dtype=np.float)
    #anatomicalMarkerVecDict['R.Shank.Upper'] -= np.array([80,0,0],dtype=np.float)
    #anatomicalMarkerVecDict['R.Shank.Front'] -= np.array([100,0,0],dtype=np.float)
    #anatomicalMarkerVecDict['R.Ankle.Lat'] -= np.array([30,0,0],dtype=np.float)
    #anatomicalMarkerVecDict['R.Ankle.Med'] -= np.array([20,0,0],dtype=np.float)
#############################


arucoDict = ar.getPredefinedDictionary(ar.DICT_4X4_50)
arucoParameters = ar.DetectorParameters_create()
arucoParameters.cornerRefinementMethod = ar.CORNER_REFINE_CONTOUR

writer = TRCWriter(list(anatomicalMarkerVecDict.keys()),fps=FPS,distination_file=DISTINATION_FILE)
writer.write_headers()

if __name__=="__main__":
    cv2.namedWindow('fig',cv2.WINDOW_FREERATIO)
    cv2.startWindowThread()
    ret, img = cap.read()
    while ret:
        #img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        corners, ids, rejected = ar.detectMarkers(img,
                                                  parameters=arucoParameters,
                                                  dictionary=arucoDict)
        trc_vec_dict = dict()
        if ids is not None:
            id_list = list(ids.reshape(-1))
            for segment_name, marker_id in markerIdDict.items():
                if marker_id in id_list:
                    r, t, trash = ar.estimatePoseSingleMarkers(corners[id_list.index(marker_id)],
                                                               cameraMatrix=mtx,
                                                               distCoeffs=dist,
                                                               markerLength=90)
                    R_marker , J = cv2.Rodrigues(r.reshape(-1))
                    t_marker = t.reshape(-1)
                    drawCube(img,90,R_marker,t_marker,mtx,dist,thickness=2,color=(255,0,255))
                    ar.drawAxis(img,mtx,dist,r,t,100)
                    tail = tuple(cv2.projectPoints(np.array([0,0,0],dtype=np.float32)
                                                   ,R_marker,t_marker,
                                                   mtx,dist)[0].reshape(-1).astype(np.int))
                    for anatomical_marker in anatomicalMarkerSegmentDict[segment_name]:
                        t_anatomic = anatomicalMarkerVecDict[anatomical_marker]
                        projected_t_anatomic = cv2.projectPoints(t_anatomic,
                                                                 R_marker,
                                                                 t_marker,
                                                                 mtx,
                                                                 dist)[0]
                        head = tuple(projected_t_anatomic.reshape(-1).astype(np.int))
                        cv2.drawMarker(img,head,thickness=4,markerSize=10,
                                       color=(255,255,0),
                                       markerType=cv2.MARKER_DIAMOND)
                        cv2.line(img,tail,head,(0,255,255),thickness=3)
                        vec = np.linalg.inv(R_ref) @ ((R_marker @ t_anatomic + t_marker) - t_ref)
                        # Write it to .trc file
                        trc_vec_dict[anatomical_marker] = vec
        writer.add_frame(trc_vec_dict)
        cv2.imshow('fig',img)
        out.write(img)
        cmd = cv2.waitKey(1)
        if (cmd & 0xff == ord('s')):
            cv2.imwrite('snapshot.jpg',img)
        ret , img = cap.read()
    writer.finish_writing()
