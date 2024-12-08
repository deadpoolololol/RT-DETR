"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

import matplotlib.pyplot as plt
from tools.visualizer import Visualizer

# # 初始化全局绘图对象
# plt.ion()
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 两个子图：训练损失和评估指标
# train_loss_values = []
# eval_mAP_values = []

# def update_train_plot(train_loss, epoch, save_dir):
#     """
#     动态更新训练损失曲线，并保存图像到指定位置
#     :param train_loss: 当前训练损失值 (张量)
#     :param epoch: 当前训练轮次
#     :param save_dir: 图像保存目录
#     """
#     train_loss_values.append(train_loss.cpu().item())  # 将张量移动到 CPU 并转换为标量
#     axs[0].clear()  # 清除当前子图
#     axs[0].plot(train_loss_values, label='Training Loss')  # 绘制训练损失曲线
#     axs[0].set_title("Training Loss")  # 设置标题
#     axs[0].set_xlabel("Iterations")  # 设置 X 轴标签
#     axs[0].set_ylabel("Loss")  # 设置 Y 轴标签
#     axs[0].legend()  # 显示图例
#     plt.pause(0.01)  # 短暂暂停以更新图表

#     # # 保存图像
#     # if save_dir:
#     #     os.makedirs(save_dir, exist_ok=True)
#     #     plt.savefig(os.path.join(save_dir, f"train_loss_epoch_{epoch}.png"))

# # 新增：更新评估指标曲线并保存图像
# def update_eval_plot(eval_mAP, epoch, save_dir):
#     """
#     动态更新评估指标曲线，并保存图像到指定位置
#     :param eval_mAP: 当前评估的 mAP 值
#     :param epoch: 当前训练轮次
#     :param save_dir: 图像保存目录
#     """
#     eval_mAP_values.append(eval_mAP)  # 记录每次的评估 mAP
#     axs[1].clear()  # 清除当前子图
#     axs[1].plot(eval_mAP_values, label='Evaluation mAP', color='orange')  # 绘制评估曲线
#     axs[1].set_title("Evaluation Metric (mAP)")  # 设置标题
#     axs[1].set_xlabel("Epochs")  # 设置 X 轴标签
#     axs[1].set_ylabel("mAP")  # 设置 Y 轴标签
#     axs[1].legend()  # 显示图例
#     plt.pause(0.01)  # 短暂暂停以更新图表

#     # 保存图像
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         # plt.savefig(os.path.join(save_dir, f"eval_mAP_epoch_{epoch}.png"))
#         plt.savefig(os.path.join(save_dir, f"eval_mAP_epoch.png"))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,visualizer: Visualizer = None, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # # 验证过滤效果 - 打印类别信息
        # filtered_categories = [t['labels'].cpu().tolist() for t in targets]
        # print(f"Filtered categories in batch: {filtered_categories}")
        

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            optimizer.zero_grad()
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # 动态更新训练损失曲线
    visualizer.update_train_plot(loss_value, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device,epoch,output_dir,visualizer):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # # 自定义目标尺寸范围 (VisDrone 标准)
    # coco_evaluator.coco_eval['bbox'].params.areaRng = [
    #     [0, 400],        # 小目标 (S)
    #     [400, 1600],     # 中目标 (M)
    #     [1600, 1e5]      # 大目标 (L)
    # ]

    # # 为了对应范围标签（areaRngLbl）也需要修改
    # coco_evaluator.coco_eval['bbox'].params.areaRngLbl = ['small', 'medium', 'large']

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]
            
    eval_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
    eval_mAP = eval_stats[0]  # 假设 mAP 是 stats 的第一个值

    # 新增：动态更新评估指标曲线
    visualizer.update_eval_plot(eval_mAP,epoch)

    return stats, coco_evaluator



