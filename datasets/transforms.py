import torch
import numpy as np

def get_rot_from_yaw(angle):
    cy=torch.cos(angle)
    sy=torch.sin(angle)
    R=torch.tensor([[cy,0,-sy],
                [0,1,0],
                [sy,0,cy]]).float()
    return R

class Aug_with_Tran(object):
    def __init__(self,jitter_surface=True,jitter_partial=True,par_jitter_sigma=0.02):
        self.jitter_surface=jitter_surface
        self.jitter_partial=jitter_partial
        self.par_jitter_sigma=par_jitter_sigma

    def __call__(self,surface,point,par_points,proj_mat,tran_mat):
        if surface is not None:surface=torch.mm(surface,tran_mat[0:3,0:3].transpose(0,1))+tran_mat[0:3,3]
        if point is not None:point=torch.mm(point,tran_mat[0:3,0:3].transpose(0,1))+tran_mat[0:3,3]
        if par_points is not None:par_points=torch.mm(par_points,tran_mat[0:3,0:3].transpose(0,1))+tran_mat[0:3,3]
        if proj_mat is not None:
            '''need to put the augmentation back'''
            inv_tran_mat = np.linalg.inv(tran_mat)
            if isinstance(proj_mat, list):
                for idx, mat in enumerate(proj_mat):
                    mat = np.dot(mat, inv_tran_mat)
                    proj_mat[idx] = mat
            else:
                proj_mat = np.dot(proj_mat, inv_tran_mat)

        if self.jitter_surface and surface is not None:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)
        if self.jitter_partial and par_points is not None:
            par_points+=self.par_jitter_sigma * torch.randn_like(par_points)


        return surface,point,par_points,proj_mat


#add small augmentation
class Scale_Shift_Rotate(object):
    def __init__(self, interval=(0.75, 1.25), angle=(-5,5), shift=(-0.1,0.1), use_scale=True,use_whole_scale=False,use_rot=True,
                 use_shift=True,jitter=True,jitter_partial=True,par_jitter_sigma=0.02,rot_shift_surface=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.angle=angle
        self.shift=shift
        self.jitter = jitter
        self.jitter_partial=jitter_partial
        self.rot_shift_surface=rot_shift_surface
        self.use_scale=use_scale
        self.use_rot=use_rot
        self.use_shift=use_shift
        self.par_jitter_sigma=par_jitter_sigma
        self.use_whole_scale=use_whole_scale

    def __call__(self, surface, point, par_points=None,proj_mat=None):
        if self.use_scale:
            scaling = torch.rand(1, 3) * 0.5 + 0.75
        else:
            scaling = torch.ones((1,3)).float()
        if self.use_shift:
            shifting = torch.rand(1,3) *(self.shift[1]-self.shift[0])+self.shift[0]
        else:
            shifting=np.zeros((1,3))
        if self.use_rot:
            angle=torch.rand(1)*(self.angle[1]-self.angle[0])+self.angle[0]
        else:
            angle=torch.tensor((0))
        #print(angle)
        angle=angle/180*np.pi
        rot_mat=get_rot_from_yaw(angle)

        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        if self.use_whole_scale:
            scale = scale*(np.random.random()*0.3+0.7)
        surface *= scale
        point *= scale

        #scale = 1

        if self.rot_shift_surface:
            surface=torch.mm(surface,rot_mat.transpose(0,1))
            surface = surface + shifting
            point=torch.mm(point,rot_mat.transpose(0,1))
            point=point+shifting

        if par_points is not None:
            par_points = par_points * scaling
            par_points=torch.mm(par_points,rot_mat.transpose(0,1))
            par_points+=shifting
            par_points *= scale

        post_scale_tran=np.eye(4)
        post_scale_tran[0,0],post_scale_tran[1,1],post_scale_tran[2,2]=scale,scale,scale
        shift_tran = np.eye(4)
        shift_tran[0:3, 3] = shifting
        rot_tran = np.eye(4)
        rot_tran[0:3, 0:3] = rot_mat
        scale_tran = np.eye(4)
        scale_tran[0, 0], scale_tran[1, 1], scale_tran[2, 2] = scaling[0, 0], scaling[
            0, 1], scaling[0, 2]

        #print(post_scale_tran,np.dot(np.dot(shift_tran,np.dot(rot_tran,scale_tran))))
        tran_mat=np.dot(post_scale_tran,np.dot(shift_tran,np.dot(rot_tran,scale_tran)))
        #tran_mat=np.dot(post_scale_tran,tran_mat)
        #print(np.linalg.norm(surface - (np.dot(org_surface,tran_mat[0:3,0:3].T)+tran_mat[0:3,3])))
        if proj_mat is not None:
            '''need to put the augmentation back'''
            inv_tran_mat=np.linalg.inv(tran_mat)
            if isinstance(proj_mat,list):
                for idx,mat in enumerate(proj_mat):
                    mat=np.dot(mat,inv_tran_mat)
                    proj_mat[idx]=mat
            else:
                proj_mat=np.dot(proj_mat,inv_tran_mat)


        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)
        if self.jitter_partial and par_points is not None:
            par_points+=self.par_jitter_sigma * torch.randn_like(par_points)

        return surface, point, par_points, proj_mat, tran_mat


class Augment_Points(object):
    def __init__(self, interval=(0.75, 1.25), angle=(-5,5), shift=(-0.1,0.1), use_scale=True,use_rot=True,
                 use_shift=True,jitter=True,jitter_sigma=0.02):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.angle=angle
        self.shift=shift
        self.jitter = jitter
        self.use_scale=use_scale
        self.use_rot=use_rot
        self.use_shift=use_shift
        self.jitter_sigma=jitter_sigma

    def __call__(self, points1,points2):
        if self.use_scale:
            scaling = torch.rand(1, 3) * 0.5 + 0.75
        else:
            scaling = torch.ones((1,3)).float()
        if self.use_shift:
            shifting = torch.rand(1,3) *(self.shift[1]-self.shift[0])+self.shift[0]
        else:
            shifting=np.zeros((1,3))
        if self.use_rot:
            angle=torch.rand(1)*(self.angle[1]-self.angle[0])+self.angle[0]
        else:
            angle=torch.tensor((0))
        #print(angle)
        angle=angle/180*np.pi
        rot_mat=get_rot_from_yaw(angle)

        points1 = points1 * scaling
        points2 = points2 * scaling

        #scale = 1
        scale = min((1 / torch.abs(points1).max().item()) * 0.999999,(1 / torch.abs(points2).max().item()) * 0.999999)
        points1 *= scale
        points2 *= scale

        points1=torch.mm(points1,rot_mat.transpose(0,1))
        points1 = points1 + shifting
        points2=torch.mm(points2,rot_mat.transpose(0,1))
        points2=points2+shifting

        if self.jitter:
            points1 += self.jitter_sigma * torch.randn_like(points1)
            points2 += self.jitter_sigma * torch.randn_like(points2)

        return points1,points2