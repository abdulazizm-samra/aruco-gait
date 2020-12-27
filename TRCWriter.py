import numpy as np
import os

class TRCWriter:
    def __init__(self,marker_names,fps=30.00,unit='mm',distination_file='output.trc'):
        self.markerNames = marker_names
        self.numMarkers = len(marker_names)
        self.FPS = fps
        self.unit = unit
        self.distinationFile = distination_file
        self.FHtemp = open(distination_file+'tmp','w')
        self.FH = open(distination_file,'w')
        self.currentFrameNum = 0
        self.ZERO_VECTOR = np.array([0,0,0],dtype=np.float)

    def write_headers(self):
        fileName = self.distinationFile.split('/')[-1]
        self.FHtemp.write(f'PathFileType	4	(X/Y/Z)	{fileName}\n')
        self.FHtemp.write('DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames\n')
        self.FHtemp.write(f'###\n')
        self.FHtemp.write('Frame#\tTime\t')
        for marker in self.markerNames:
            self.FHtemp.write(f'{marker}\t\t\t')
        self.FHtemp.write('\n\t\t')
        for i in range(1,self.numMarkers+1):
            self.FHtemp.write(f'X{i}\tY{i}\tZ{i}\t')
        self.FHtemp.write('\n\n')

    def add_frame(self,data_dict:dict):
        self.currentFrameNum = self.currentFrameNum + 1
        self.FHtemp.write(f'{self.currentFrameNum}\t{(self.currentFrameNum-1)/self.FPS}\t')
        for marker in self.markerNames:
            vec = data_dict.get(marker,self.ZERO_VECTOR)
            self.FHtemp.write(f'{vec[0]}\t{vec[1]}\t{vec[2]}\t')
        self.FHtemp.write('\n')

    def finish_writing(self):
        self.FHtemp.close()
        self.FHtemp = open(self.distinationFile+'tmp','r')
        for line in self.FHtemp:
            if line.startswith('###'):
                self.FH.write(
                    f'{self.FPS}\t{self.FPS}\t{self.currentFrameNum}\t{self.numMarkers}\t{self.unit}\t{self.FPS}\t1\t{self.currentFrameNum}\n'
                )
            else:
                self.FH.write(line)
        self.FH.close()

if __name__=="__main__":
    x = np.array([0,1,2],dtype=np.float32)
    writer = TRCWriter(['a','b'])
    writer.write_headers()
    writer.add_frame({'a':   x,'b':-2*x})
    writer.add_frame({'a': 3*x,'b': 2*x})
    writer.add_frame({'a':-1*x})
    writer.add_frame({'a': 5*x,'b':2*x})
    writer.add_frame({'a':-4*x})
    writer.finish_writing()








