import torch.utils.data

from .Multiview_dataset import Object_Occ,Object_PartialPoints_MultiImg
from .transforms import Scale_Shift_Rotate,Aug_with_Tran, Augment_Points
from .taxonomy import synthetic_category_combined,synthetic_arkit_category_combined,arkit_category

def build_object_occ_dataset(split,args):
    transform = Scale_Shift_Rotate(rot_shift_surface=True,use_scale=True,use_whole_scale=True)
    category=args['category']
    #category_list=synthetic_category_combined[category]
    category_list=synthetic_arkit_category_combined[category]
    replica=args['replica']
    if split == "train":
        return Object_Occ(args['data_path'], split=split, categories=category_list,
                                        transform=transform, sampling=True,
                                        num_samples=args['num_samples'], return_surface=True,
                                        surface_sampling=True, surface_size=args['surface_size'],replica=replica)
    elif split == "val":
        return Object_Occ(args['data_path'], split=split,categories=category_list,
                                        transform=transform, sampling=False,
                                        num_samples=args['num_samples'], return_surface=True,
                                        surface_sampling=True,surface_size=args['surface_size'], replica=1)

def build_par_multiimg_dataset(split,args):
    #transform=Scale_Shift_Rotate(rot_shift_surface=False,use_scale=False,use_shift=False,use_rot=False) #fix the encoder into cannonical space
    #transform=Scale_Shift_Rotate(rot_shift_surface=True)
    transform=Aug_with_Tran(par_jitter_sigma=args['jitter_partial_train'])
    val_transform=Aug_with_Tran(par_jitter_sigma=args['jitter_partial_val'])
    category=args['category']
    category_list=synthetic_category_combined[category]
    if split == "train":
        return Object_PartialPoints_MultiImg(args['data_path'], split_filename="train_par_img.json",split=split,
                                categories=category_list,
                                transform=transform, sampling=True,
                                num_samples=1024, return_surface=False,ret_sample=False,
                                surface_sampling=True, par_pc_size=args['par_pc_size'],surface_size=args['surface_size'],
                                load_proj_mat=args['load_proj_mat'],load_image=args['load_image'],load_triplane=True,
                                par_prefix=args['par_prefix'],par_point_aug=args['par_point_aug'],replica=args['replica'],
                                             num_objects=args['num_objects'])
    elif split  =="val":
        return Object_PartialPoints_MultiImg(args['data_path'], split_filename="val_par_img.json",split=split,
                                categories=category_list,
                                transform=val_transform, sampling=False,
                                num_samples=1024, return_surface=False,ret_sample=True,
                                surface_sampling=True, par_pc_size=args['par_pc_size'],surface_size=args['surface_size'],
                                load_proj_mat=args['load_proj_mat'],load_image=args['load_image'],load_triplane=True,
                                par_prefix=args['par_prefix'],par_point_aug=None,replica=1)

def build_finetune_par_multiimg_dataset(split,args):
    #transform=Scale_Shift_Rotate(rot_shift_surface=False,use_scale=False,use_shift=False,use_rot=False) #fix the encoder into cannonical space
    #transform=Scale_Shift_Rotate(rot_shift_surface=True)
    keyword=args['keyword']
    pretrain_transform=Aug_with_Tran(par_jitter_sigma=args['jitter_partial_pretrain']) #add more noise to partial points
    finetune_transform=Aug_with_Tran(par_jitter_sigma=args['jitter_partial_finetune'])
    val_transform=Aug_with_Tran(par_jitter_sigma=args['jitter_partial_val'])

    pretrain_cat=synthetic_category_combined[args['category']]
    arkit_cat=arkit_category[args['category']]
    use_pretrain_data=args["use_pretrain_data"]
    #print(arkit_cat,pretrain_cat)
    if split == "train":
        if use_pretrain_data:
            pretrain_dataset=Object_PartialPoints_MultiImg(args['data_path'], split_filename="train_par_img.json",categories=pretrain_cat,
                                    split=split,transform=pretrain_transform, sampling=True,num_samples=1024, return_surface=False,ret_sample=False,
                                    surface_sampling=True, par_pc_size=args['par_pc_size'],surface_size=args['surface_size'],
                                    load_proj_mat=args['load_proj_mat'],load_image=args['load_image'],load_triplane=True,par_point_aug=args['par_point_aug'],
                                                           par_prefix=args['par_prefix'],replica=1)
        finetune_dataset=Object_PartialPoints_MultiImg(args['data_path'], split_filename=keyword+"_train_par_img.json",categories=arkit_cat,
                                split=split,transform=finetune_transform, sampling=True,num_samples=1024, return_surface=False,ret_sample=False,
                                surface_sampling=True, par_pc_size=args['par_pc_size'],surface_size=args['surface_size'],
                                load_proj_mat=args['load_proj_mat'],load_image=args['load_image'],load_triplane=True,par_point_aug=None,replica=args['replica'])
        if use_pretrain_data:
            return torch.utils.data.ConcatDataset([pretrain_dataset,finetune_dataset])
        else:
            return finetune_dataset
    elif split  =="val":
        return Object_PartialPoints_MultiImg(args['data_path'], split_filename=keyword+"_val_par_img.json",categories=arkit_cat,split=split,
                                transform=val_transform, sampling=False,
                                num_samples=1024, return_surface=False,ret_sample=True,
                                surface_sampling=True, par_pc_size=args['par_pc_size'],surface_size=args['surface_size'],
                                load_proj_mat=args['load_proj_mat'],load_image=args['load_image'],load_triplane=True,par_point_aug=None,replica=1)

def build_dataset(split,args):
    if args['type']=="Occ":
        return build_object_occ_dataset(split,args)
    elif args['type']=="Occ_Par_MultiImg":
        return build_par_multiimg_dataset(split,args)
    elif args['type']=="Occ_Par_MultiImg_Finetune":
        return build_finetune_par_multiimg_dataset(split,args)
    else:
        raise NotImplementedError