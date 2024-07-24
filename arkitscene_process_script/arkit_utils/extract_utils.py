import json
import numpy as np
from .taxonomy import class_names
from .box_utils import compute_box_3d,get_size,generate_bbox_voxel

def extract_gt(gt_fn):
    """extract original label data

    Args:
        gt_fn: str (file name of "annotation.json")
            after loading, we got a dict with keys
                'data', 'stats', 'comment', 'confirm', 'skipped'
            ['data']: a list of dict for bboxes, each dict has keys:
                'uid', 'label', 'modelId', 'children', 'objectId',
                'segments', 'hierarchy', 'isInGroup', 'labelType', 'attributes'
                'label': str
                'segments': dict for boxes
                    'centroid': list of float (x, y, z)?
                    'axesLengths': list of float (x, y, z)?
                    'normalizedAxes': list of float len()=9
                'uid'
            'comments':
            'stats': ...
    Returns:
        skipped: bool
            skipped or not
        boxes_corners: (n, 8, 3) box corners
            **world-coordinate**
        centers: (n, 3)
            **world-coordinate**
        sizes: (n, 3) full-sizes (no halving!)
        labels: list of str
        uids: list of str
    """
    gt = json.load(open(gt_fn, "r"))
    skipped = gt['skipped']
    if len(gt['data']) == 0:
        boxes_corners = np.zeros((0, 8, 3))
        centers = np.zeros((0, 3))
        sizes = np.zeros((0, 3))
        labels, uids = [], []
        bbox_voxels=np.zeros((0,128))
        boxes_mat=np.zeros((0,4,4))
        return skipped, boxes_corners, centers, sizes, labels, uids,bbox_voxels,boxes_mat

    boxes_corners = []
    centers = []
    sizes = []
    labels = []
    uids = []
    bbox_voxels=[]
    boxes_mat=[]
    for data in gt['data']:
        l = data["label"]
        for delimiter in [" ", "-", "/"]:
            l = l.replace(delimiter, "_")
        if l not in class_names:
            print("unknown category: %s" % l)
            continue

        rotmat = np.array(data["segments"]["obbAligned"]["normalizedAxes"]).reshape(
            3, 3
        )
        center = np.array(data["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        size = np.array(data["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
        #object coordinate is -1 ~ 1
        box_mat=np.eye(4)
        size_mat=np.eye(4)
        max_size=np.amax(size)
        #size_mat[0,0],size_mat[1,1],size_mat[2,2]=max_size/2,max_size/2,max_size/2 #do not need scale yet
        shift_mat=np.eye(4)
        shift_mat[0:3,3]=center
        rotmat4=np.eye(4)
        rotmat4[0:3,0:3]=rotmat
        box_mat=np.dot(size_mat,box_mat)
        box_mat=np.dot(rotmat4.T,box_mat)
        box_mat=np.dot(shift_mat,box_mat)
        boxes_mat.append(box_mat[:,:])
        # Padding bbox
        #size *= 1.2
        box3d = compute_box_3d(size.reshape(3).tolist(), center, rotmat)
        # box3d_canon=compute_box_3d([2.0,2.0,2.0],[0,0,0],np.eye(3))
        # box3d_canon_homo=np.concatenate([box3d_canon,np.ones((box3d_canon.shape[0],1))],axis=1)
        # box3d_recover=np.dot(box3d_canon_homo,box_mat.T)[:,0:3]
        #print("orginal",box3d)
        #print("recover",box3d_recover)


        bbox_vox=generate_bbox_voxel(size.reshape(3).tolist(),center,rotmat)
        '''
            Box corner order that we return is of the format below:
                6 -------- 7
               /|         /|
              5 -------- 4 .
              | |        | |
              . 2 -------- 3
              |/         |/
              1 -------- 0 
        '''
        boxes_corners.append(box3d.reshape(1, 8, 3))
        size = np.array(get_size(box3d)).reshape(1, 3)
        center = np.mean(box3d, axis=0).reshape(1, 3)
        bbox_vox=bbox_vox.reshape(1,1000,3)
        bbox_voxels.append(bbox_vox)
        # boxes_corners.append(box3d.reshape(1, 8, 3))
        centers.append(center)
        sizes.append(size)
        # labels.append(l)
        labels.append(data["label"])
        uids.append(data["uid"])
    centers = np.concatenate(centers, axis=0)
    sizes = np.concatenate(sizes, axis=0)
    boxes_corners = np.concatenate(boxes_corners, axis=0)
    bbox_voxels=np.concatenate(bbox_voxels,axis=0)
    boxes_mat=np.stack(boxes_mat)
    return skipped, boxes_corners, centers, sizes, labels, uids,bbox_voxels, boxes_mat