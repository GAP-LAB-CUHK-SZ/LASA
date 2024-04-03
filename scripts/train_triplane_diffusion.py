import argparse
import sys
sys.path.append("..")
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models import get_model,get_criterion

from engine.engine_triplane_dm import train_one_epoch,evaluate_reconstruction

def get_args_parser():
    parser = argparse.ArgumentParser('Latent Diffusion', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--ae-pth',type=str)
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',  # 2e-4
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    # Dataset parameters
    parser.add_argument('--data-pth', default='../data', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=60, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--constant_lr', default=False, action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--load_proj_mat',default=True,type=bool)
    parser.add_argument('--num_objects',type=int,default=-1)

    parser.add_argument('--configs', type=str)
    parser.add_argument('--finetune', default=False, action="store_true")
    parser.add_argument('--finetune-pth', type=str)
    parser.add_argument('--use_cls_free',action="store_true",default=False)
    parser.add_argument('--sync_bn',action="store_true",default=False)
    parser.add_argument('--category',type=str)
    parser.add_argument('--stop',type=int,default=1000)
    parser.add_argument('--replica', type=int, default=5)

    return parser


def main(args,config):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_config = config.config['dataset']
    dataset_config['category']=args.category
    dataset_config['replica']=args.replica
    dataset_config['num_objects']=args.num_objects
    dataset_config['data_path']=args.data_pth
    dataset_train = build_dataset('train', dataset_config)
    print("training dataset len is %d"%(len(dataset_train)))
    dataset_val=build_dataset('val', dataset_config)
    #dataset_val = build_dataset('val', dataset_config)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if misc.get_rank()==0:
        log_dir=args.log_dir
        src_folder="/data1/haolin/TriplaneDiffusion"
        misc.log_codefiles(src_folder,log_dir+"/code_bak")
        #cmd="cp -r %s %s"%(src_folder,log_dir+"/code_bak")
        #print(cmd)
        #os.system(cmd)
        config_dict=vars(args)
        config_save_path=os.path.join(log_dir,"config.json")
        with open(config_save_path,'w') as f:
            json.dump(config_dict,f,indent=4)

        model_dict=config
        model_config_save_path=os.path.join(log_dir,"model.json")
        config.write_config(model_config_save_path)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=args.batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    ae_args=config.config['model']['ae']
    ae = get_model(ae_args)
    ae.eval()
    print("Loading autoencoder %s" % args.ae_pth)
    ae.load_state_dict(torch.load(args.ae_pth, map_location='cpu')['model'])
    ae.to(device)

    dm_args=config.config['model']['dm']
    if args.category[0] == "all":
        dm_args["use_cat_embedding"]=True
    else:
        dm_args["use_cat_embedding"] = False
    dm_model = get_model(dm_args)
    if args.sync_bn:
        dm_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dm_model)
    if args.finetune:
        print("finetune the model, load from %s"%(args.finetune_pth))
        dm_model.load_state_dict(torch.load(args.finetune_pth,map_location="cpu")['model'])
    dm_model.to(device)

    model_without_ddp = dm_model
    n_parameters = sum(p.numel() for p in dm_model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        dm_model = torch.nn.parallel.DistributedDataParallel(dm_model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = dm_model.module

    # # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    cri_args=config.config['criterion']
    criterion = get_criterion(cri_args)

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, dm_model, device)
        print(f"loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = 1000.0
    max_iou=0

    stop_epochs=min(args.stop,args.epochs)
    for epoch in range(args.start_epoch, stop_epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        #test_stats = evaluate_reconstruction(data_loader_val, dm_model, ae, criterion, device)
        train_stats = train_one_epoch(
            dm_model, ae, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            log_dir=args.log_dir,
            args=args
        )
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=dm_model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch,prefix="latest")

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            test_stats = evaluate_reconstruction(data_loader_val, dm_model, ae, criterion, device)
            print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
            # print(f"loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}")

            if test_stats["iou"] > max_iou:
                max_iou = test_stats["iou"]
                misc.save_model(
                    args=args, model=dm_model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, prefix='best')
            else:
                misc.save_model(
                    args=args, model=dm_model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, prefix='latest')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
                log_writer.add_scalar('perf/test_iou', test_stats['iou'], epoch)
                log_writer.add_scalar('perf/test_accuracy', test_stats['accuracy'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config_path = args.configs
    from configs.config_utils import CONFIG

    config = CONFIG(config_path)
    main(args,config)
