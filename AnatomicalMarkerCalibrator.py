import cv2
import numpy as np
import sys
import cv2.aruco as ar
import pickle

from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi

class AMC(QDialog):
    def __init__(self):
        super(AMC, self).__init__()
        loadUi('anatomical marker calibrator.ui',self)
        self.anatomicalMarkers = [
            'select anatomical marker',
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
        for marker in self.anatomicalMarkers:
            self.anatomicalMarkerBox.addItem(marker)
        self.anatomicalMarkerVec = dict()
        self.addButton.clicked.connect(self.add_marker)
        self.loadCameraButton.clicked.connect(self.load_camera_file)
        self.loadRodButton.clicked.connect(self.load_rod_file)
        self.pauseButton.clicked.connect(self.pause_streaming)
        self.resumeButton.clicked.connect(self.resume_streaming)
        self.saveButton.clicked.connect(self.save_all)
        self.resetButton.clicked.connect(self.reset_all)
        self.startButton.clicked.connect(self.start_streaming)
        self.browseButton.clicked.connect(self.get_video_source)
        self.speedSlider.sliderReleased.connect(self.adjustFPS)
        self.pelvisSpin.valueChanged.connect(self.change_marker_ids)
        self.femurRSpin.valueChanged.connect(self.change_marker_ids)
        self.femurLSpin.valueChanged.connect(self.change_marker_ids)
        self.tibiaRSpin.valueChanged.connect(self.change_marker_ids)
        self.tibiaLSpin.valueChanged.connect(self.change_marker_ids)
        self.lengthSpin.valueChanged.connect(self.change_marker_length)
        self.change_marker_length()
        self.markerIdDict = dict()
        self.speedSlider.setValue(50)
        self.delay = 20
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.init_aruco_detector()
        self.logPrint('Ready')

    @pyqtSlot()
    def save_all(self):
        try:
            distination_path , filters = QFileDialog.getSaveFileName(self,directory='result.pkl')
            all_data_dict = {
                'marker_ids' : self.markerIdDict,
                'anatomical_marker_vec' : self.anatomicalMarkerVec
            }
            with open(distination_path,'wb') as fh:
                pickle.dump(all_data_dict,fh)

            self.logPrint(f'Saved to :{distination_path}')
        except Exception as e:
            self.logPrint(e)

    @pyqtSlot()
    def reset_all(self):
        try:
            self.anatomicalMarkerVec.clear()
            self.markerIdDict.clear()
            self.logPrint("Anatomical marker vectors and marker id's was cleared!")
        except Exception as e:
            self.logPrint(e)

    @pyqtSlot()
    def change_marker_length(self):
        self.markerLength = self.lengthSpin.value()

    @pyqtSlot()
    def change_marker_ids(self):
        try:
            self.markerIdDict['pelvis'] = self.pelvisSpin.value()
            self.markerIdDict['femur_r'] = self.femurRSpin.value()
            self.markerIdDict['femur_l'] = self.femurLSpin.value()
            self.markerIdDict['tibia_r'] = self.tibiaRSpin.value()
            self.markerIdDict['tibia_l'] = self.tibiaLSpin.value()
        except Exception as e:
            self.logPrint(e)

    @pyqtSlot()
    def add_marker(self):
        try:
            corners, ids, rejected = ar.detectMarkers(self.img, self.arucoDict,parameters=self.arucoParameters)
            id_list = list(ids.reshape(-1))
            target_marker_name = self.anatomicalMarkerBox.currentText().split('/')
            if len(target_marker_name)==2:
                segment_name = target_marker_name[0]
                target_marker_name = target_marker_name[1]
                segment_id = self.markerIdDict[segment_name]
                R_rod , t_rod , R_seg , t_seg = None, None, None, None
                if self.rodMarkerID in id_list:
                    r, t, trash = ar.estimatePoseSingleMarkers(
                        corners[id_list.index(self.rodMarkerID)],
                        markerLength=self.markerLength,
                        cameraMatrix=self.mtx,
                        distCoeffs=self.dist
                    )
                    t_rod = t.reshape(-1)
                    R_rod, J = cv2.Rodrigues(r)
                else:
                    self.logPrint("Rod marker doesn't appear in the image!")

                if segment_id in id_list:
                    r, t, trash = ar.estimatePoseSingleMarkers(
                        corners[id_list.index(segment_id)],
                        markerLength=self.markerLength,
                        cameraMatrix=self.mtx,
                        distCoeffs=self.dist
                    )
                    t_seg = t.reshape(-1)
                    R_seg, J = cv2.Rodrigues(r)
                else:
                    self.logPrint('Body segment marker doesn`t appear in the image!')

                if R_seg is not None and R_rod is not None:
                    point_seg = np.linalg.inv(R_seg) @ ((R_rod @ self.rodMarkerTVec + t_rod) - t_seg)
                    self.anatomicalMarkerVec[target_marker_name] = point_seg
                    self.logPrint('Anatomical marker was add :')
                    self.logPrint(f'{target_marker_name} = {point_seg}')

            else:
                self.logPrint('Please select the marker...')

        except Exception as e:
            self.logPrint(e)

    @pyqtSlot()
    def load_camera_file(self):
        path , filters = QFileDialog.getOpenFileName(self, '')
        try:
            file = np.load(path)
            self.mtx = file['mtx']
            self.dist = file['dist']
            self.logPrint('Camera coefficients loaded successfully' )
            self.logPrint(f'mtx = {self.mtx}')
            self.logPrint(f'dist = {self.dist}')
        except Exception as e:
            self.logPrint(e)

    @pyqtSlot()
    def load_rod_file(self):
        path , filters = QFileDialog.getOpenFileName(self, '')
        try:
            file = np.load(path)
            self.rodMarkerID = file['marker_id']
            self.rodMarkerTVec = file['tvec']
            self.logPrint('Rod marker coefficients loaded successfully' )
            self.logPrint(f'marker_id = {self.rodMarkerID}')
            self.logPrint(f'tvec = {self.rodMarkerTVec}')
        except Exception as e:
            self.logPrint(e)

    @pyqtSlot()
    def get_video_source(self):
        self.VIDEO_PATH , filters = QFileDialog.getOpenFileName(self, '')
        self.sourcePathEdit.setText(self.VIDEO_PATH)

    @pyqtSlot()
    def start_streaming(self):
        try:
            source = self.sourcePathEdit.text()
            self.cap = cv2.VideoCapture(source)
            self.timer.start(self.delay)
        except Exception as e:
            self.logPrint(e)
            self.timer.stop()

    @pyqtSlot()
    def pause_streaming(self):
        self.timer.stop()

    @pyqtSlot()
    def resume_streaming(self):
        self.timer.start(self.delay)

    @pyqtSlot()
    def adjustFPS(self):
        s = self.speedSlider.value()
        self.delay = int (1000 / s)
        self.timer.stop()
        self.timer.start(self.delay)

    def init_aruco_detector(self):
        self.arucoDict = ar.getPredefinedDictionary(ar.DICT_4X4_50)
        self.arucoParameters = ar.DetectorParameters_create()
        self.arucoParameters.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX

    def update_image(self):
        try:
            if (self.VIDEO_PATH.endswith('jpg')):
                self.img = cv2.imread(self.VIDEO_PATH)
            else:
                _, self.img = self.cap.read()
            if self.rotateBox.isChecked():
                self.img = cv2.rotate(self.img,cv2.ROTATE_90_CLOCKWISE)
            self.processed_image = self.process_image(self.img.copy())
            self.displayImage(self.processed_image,self.imageLabel)
        except Exception as e:
            self.logPrint(e)
            self.timer.stop()

    def process_image(self,img):
        if self.enableDetection.isChecked() == True:
            self.corners,self.ids,rejected = ar.detectMarkers(
                img,self.arucoDict
            )
            ar.drawDetectedMarkers(img,self.corners,self.ids)
            for id, corners in zip(self.ids,self.corners):
                r, t, trash = ar.estimatePoseSingleMarkers(corners,
                                                           self.markerLength,
                                                           self.mtx,self.dist)
                ar.drawAxis(img,self.mtx,self.dist,r,t,self.markerLength)
                if id == self.rodMarkerID:
                    rod_vertices = list(
                        cv2.projectPoints(
                            np.array([np.zeros(3),self.rodMarkerTVec]),
                                      r,t,
                                      self.mtx,self.dist)[0].reshape(-1,2).astype(np.int)
                        )
                    cv2.line(img,
                             pt1 = tuple(rod_vertices[0]),
                             pt2 = tuple(rod_vertices[1]),
                             color=(255,0,255),
                             thickness=3)
        return img

    def logPrint(self,text):
        self.logBrowser.append(str(text))

    def displayImage(self, image, label):
        qformat = QImage.Format_Indexed8  # Grayscale
        if (len(image.shape) == 3):  # there are Channels
            if (image.shape[2] == 4):  # RGBA
                qformat = QImage.Format_RGBA8888
            elif (image.shape[2] == 3):  # RGB
                qformat = QImage.Format_RGB888

        img = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        img = img.rgbSwapped()
        label.setPixmap(QPixmap.fromImage(img))


if __name__=="__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = AMC()
    window.show()
    sys.exit(app.exec_())