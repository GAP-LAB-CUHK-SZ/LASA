import torch
import hat.archs
import hat.data
import hat.models
from basicsr.models import build_model
from basicsr.utils.options import dict2str, parse_options
import os.path as osp
from PIL import Image
import numpy as np
import glob
import os,sys
sys.path.append("..")
import tqdm
from torch.utils.data import Dataset, DataLoader
import builtins
import datetime
import torch.distributed as dist
import yaml
from collections import OrderedDict
import misc


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

class simple_image_dataset(Dataset):
    def __init__(self,arkit_image_dir,valid_list_dir=None):
        scene_list = os.listdir(arkit_image_dir)
        self.valid_list_dir=valid_list_dir
        self.image_path_list=[]
        self.basename_list = []
        self.object_id_list=[]
        self.scene_id_list=[]
        for scene_id in scene_list:
            scene_folder=os.path.join(arkit_image_dir,scene_id)
            object_id_list=os.listdir(scene_folder)
            for object_id in object_id_list:
                color_folder = os.path.join(scene_folder,object_id, "color")
                filelist=os.listdir(color_folder)
                image_id_list=[filename[:-4] for filename in filelist]
                for image_id in image_id_list:
                    color_filepath=os.path.join(color_folder,"%s.png"%(image_id))
                    self.image_path_list.append(color_filepath)
                    self.basename_list.append(os.path.basename(color_filepath))
                    self.scene_id_list.append(scene_id)
                    self.object_id_list.append(object_id)
        #print(len(self.image_path_list))
    def __len__(self):
        return len(self.image_path_list)
    def __getitem__(self,index):
        #print("start accessing data")
        image_path=self.image_path_list[index]
        basename=self.basename_list[index]
        scene_id=self.scene_id_list[index]
        object_id=self.object_id_list[index]

        image = np.asarray(Image.open(image_path)) / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        #print("load image")

        return image,basename,scene_id,object_id

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--opt",type=str,default="./HAT_SRx4_ImageNet-pretrain.yml")
parser.add_argument("--image_dir",type=str,required=True)

parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
args=parser.parse_args()

misc.init_distributed_mode(args)

opt=yaml_load(args.opt)
opt['is_train']=False
opt['dist']=False
opt['path']['pretrain_network_g']='../checkpoint/SR_model/HAT_SRx4_ImageNet-pretrain.pth'

print("building model")
model = build_model(opt)

arkit_image_dir=args.image_dir
save_dir=arkit_image_dir
dataset=simple_image_dataset(arkit_image_dir,None)
num_tasks = misc.get_world_size()
local_rank = misc.get_rank()
sampler = torch.utils.data.DistributedSampler(
    dataset, num_replicas=num_tasks, rank=local_rank,
    shuffle=False)  # shuffle=True to reduce monitor bias

print("building dataloader")
dataloader=DataLoader(
    dataset=dataset,
    sampler=sampler,
    num_workers=4,
    pin_memory=False,
    batch_size=4,
    shuffle=False
)
length=len(dataloader)
print("start super resolution on images")
for idx,item in enumerate(dataloader):
    image,basename_list,scene_id_list,object_id_list=item
    print("%d/%d, rank %d" % (idx, length, local_rank), "processing",set(scene_id_list))
    image=image.float().cuda()
    data_dict = {
        "lq":image
    }
    with torch.no_grad():
        model.feed_data(data_dict)
        model.test()
    highres_image=model.output
    highres_image = torch.clamp(highres_image, max=1.0, min=0.0)
    highres_image=highres_image.permute(0,2,3,1).detach().cpu().numpy()
    highres_image=(highres_image*255.0).astype(np.uint8)
    for i in range(highres_image.shape[0]):
        highres_image_pil=Image.fromarray(highres_image[i])
        basename=basename_list[i]
        save_folder=os.path.join(save_dir,scene_id_list[i],object_id_list[i],"highres_color")
        os.makedirs(save_folder,exist_ok=True)
        save_path=os.path.join(save_folder,basename.replace(".png",".jpg"))
        highres_image_pil.save(save_path)
