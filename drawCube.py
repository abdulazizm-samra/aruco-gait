import cv2
import numpy as np

def drawCube(img,length,r,t,mtx,dist,color=(0,0,255),thickness=1):
    img1 = img
    points = np.array(
        [
            [-0.5, -0.5, 0],
            [ 0.5, -0.5, 0],
            [ 0.5,  0.5, 0],
            [-0.5,  0.5, 0],
            [-0.5, -0.5, 1],
            [ 0.5, -0.5, 1],
            [ 0.5,  0.5, 1],
            [-0.5,  0.5, 1],
        ],dtype=np.float32
    ).T * length

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    image_points = list(
        cv2.projectPoints(
            points,r,t,mtx,dist
        )[0].reshape(-1,2).astype(np.int)
    )

    for i,j in edges:
        img1 = cv2.line(img1,
                        tuple(image_points[i]),tuple(image_points[j])
                        ,color,thickness)

    return img1
