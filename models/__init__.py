from .TriplaneVAE import TriplaneVAE
from .Triplane_Diffusion import Triplane_Diff_MultiImgCond_EDM
from .Triplane_Diffusion import EDMLoss_MultiImgCond
#from .Point_Diffusion_EDM import PointEDM,EDMLoss_PointAug

def get_model(model_args):
    if model_args['type']=="TriVAE":
        model=TriplaneVAE(model_args)
    elif model_args['type']=="triplane_diff_multiimg_cond":
        model=Triplane_Diff_MultiImgCond_EDM(model_args)
    else:
        raise NotImplementedError
    return model

def get_criterion(cri_args):
    if cri_args['type']=="EDMLoss_MultiImgCond":
        criterion=EDMLoss_MultiImgCond(use_par=cri_args['use_par'])
    else:
        raise NotImplementedError
    return criterion
