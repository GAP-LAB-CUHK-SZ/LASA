# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
import os
import pickle as p
import torch.distributed as dist
import time
from models.modules.encoder import DiagonalGaussianDistribution


def train_one_epoch(model: torch.nn.Module, ae: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,log_dir=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    use_cls_free= args.use_cls_free

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data_batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if not args.constant_lr:
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        input_dict=model.module.prepare_data(data_batch)
        with torch.cuda.amp.autocast(enabled=False):
            loss_all = criterion(model,input_dict,classifier_free=use_cls_free)
            loss=loss_all.mean()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_reconstruction(data_loader, model, ae, criterion, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for data_batch in metric_logger.log_every(data_loader, 50, header):
        with torch.no_grad():
            input_dict=model.module.prepare_data(data_batch)
            loss_all = criterion(model, input_dict,classifier_free=False)
            loss = loss_all.mean()
            sample_input=model.module.prepare_sample_data(data_batch)
            sampled_array = model.module.sample(sample_input).float()
            sampled_array = torch.nn.functional.interpolate(sampled_array, scale_factor=2, mode="bilinear")
            eval_input=model.module.prepare_eval_data(data_batch)
            samples=eval_input["samples"]
            labels=eval_input["labels"]
            for j in range(sampled_array.shape[0]):
                output = ae.decode(sampled_array[j:j + 1], samples[j:j+1]).squeeze(-1)
                pred = torch.zeros_like(output)
                pred[output >= 0.0] = 1
                label=labels[j:j+1]

                accuracy = (pred == label).float().sum(dim=1) / label.shape[1]
                accuracy = accuracy.mean()
                intersection = (pred * label).sum(dim=1)
                union = (pred + label).gt(0).sum(dim=1)
                iou = intersection * 1.0 / union + 1e-5
                iou = iou.mean()

                metric_logger.update(iou=iou.item())
                metric_logger.update(accuracy=accuracy.item())
        metric_logger.update(loss=loss.item())
    metric_logger.synchronize_between_processes()
    print('* iou {ious.global_avg:.3f}'
          .format(ious=metric_logger.iou))
    print('* accuracy {accuracies.global_avg:.3f}'
          .format(accuracies=metric_logger.accuracy))
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}