import numpy as np
import cv2
def draw_proj_image(image,proj_mat,points):
    points_homo=np.concatenate([points,np.ones((points.shape[0],1))],axis=1)
    pts_inimg=np.dot(points_homo,proj_mat.T)
    image=cv2.resize(image,dsize=(224,224),interpolation=cv2.INTER_LINEAR)
    x=pts_inimg[:,0]/pts_inimg[:,2]
    y=pts_inimg[:,1]/pts_inimg[:,2]
    x=np.clip(x,a_min=0,a_max=223).astype(np.int32)
    y=np.clip(y,a_min=0,a_max=223).astype(np.int32)
    image[y,x]=np.array([[0,255,0]])
    return image