import numpy as np
import matplotlib.pyplot as plt

src = 'IK_body/result.trc'
fh = open(src,'r')

next(fh)
next(fh)
line = fh.readline()
s_line = line.strip().split()

num_frames = int(s_line[2])
num_markers = int(s_line[3])

A = np.zeros((num_markers*3,num_frames),dtype=np.float)
j = 0
next(fh)
next(fh)
for line in fh:
    s_line = line.strip().split()[2:]
    print(len(s_line))
    if len(s_line)!=num_markers*3:
        continue

    for i in range(len(s_line)):
        A[i,j] = float(s_line[i])
    j = j+1

B = np.zeros((num_markers*3,num_frames),dtype=np.float)
for i in range(num_markers*3):
    B[i,:] = np.convolve(A[i,],np.ones((10,))*0.1,mode='same')

fig0 , axes0 = plt.subplots()

for i in range(num_markers*3):
    axes0.plot(np.arange(0,num_frames),A[i,:])
fig0.show()

fig1 , axes1= plt.subplots()

for i in range(num_markers*3):
    axes1.plot(np.arange(0,num_frames),B[i,:])
fig1.show()

fh = open(src,'r')
fh2 = open('filtered.trc','w')
j = 0
for line in fh:
    s_line = line.strip().split()
    if len(s_line)>0 and s_line[0].isnumeric():
        fh2.write(f'{s_line[0]}\t{s_line[1]}\t')
        for i in range(num_markers*3):
            fh2.write(f'{B[i,j]}\t')
        j = j + 1
        fh2.write('\n')
    else:
        fh2.write(line)

plt.pause(100)
