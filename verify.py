import os
import sys
import numpy
from datetime import datetime
from typing import Dict
import numpy as np
import time

import torch.nn as nn
import pandas as pd
import monai
import pytz
import torch
import yaml
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate.tracking import GeneralTracker
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
from safetensors.torch import load_model, save_model
from src import utils
from src.models import give_model
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model, MyCustomTracker
from src.loss import KL_divergence

import warnings

warnings.filterwarnings('ignore')


@torch.no_grad()
def val_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                  val_loader: torch.utils.data.DataLoader,
                  config: EasyDict, metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                  post_trans: monai.transforms.Compose, accelerator: Accelerator):
    # 验证
    model.eval()
    for i, image_batch in enumerate(val_loader):

        if config.finetune.model_choose == "MoSID":
            modal_TC, modal_VC, modal_VG, recon_TC_VC, recon_TC, recon_VC, all_true = image_batch

            InforVG = recon_TC_VC[0]
            InforTC = torch.abs(recon_TC[0] - all_true[0])
            InforVC = torch.abs(recon_VC[0] - all_true[0])
        else:
            modal_TC, modal_VC, modal_VG = image_batch

        # modal_TC, modal_VC, modal_VG = image_batch

        for m in config.dataset.Breast_US.modal_mask:
            if m == "TC":
                modal_TC[0] = torch.ones_like(modal_TC[0])
            if m == "VC":
                modal_VC[0] = torch.ones_like(modal_VC[0])
            if m == "VG":
                modal_VG[0] = torch.ones_like(modal_VG[0])

        gt = modal_VG[1]

        if config.finetune.model_choose == "mmFormer" or config.finetune.model_choose == "RFNet":
            logits = model(modal_TC[0], modal_VC[0], modal_VG[0], is_training=False)
            log = ''
            total_loss = loss_functions['dice_loss'](logits, gt)
            # accelerator.log({'Val/' + 'dice_loss': float(total_loss)}, step=step)
            val_outputs = post_trans(logits)



        elif config.finetune.model_choose == "NestedFormer" or config.finetune.model_choose == "MMEFUNet" or config.finetune.model_choose == "MMCANET":
            logits = model(modal_TC[0], modal_VC[0], modal_VG[0])
            log = ''
            total_loss = loss_functions['dice_loss'](logits, gt)
            # accelerator.log({'Val/' + 'dice_loss': float(total_loss)}, step=step)
            val_outputs = post_trans(logits)



        elif config.finetune.model_choose == "MAML":
            logits = model(modal_TC[0], modal_VC[0], modal_VG[0], is_training=False)
            log = ''
            total_loss = loss_functions['dice_ce_loss'](logits, gt)
            # accelerator.log({'Val/' + 'dice_ce_loss': float(total_loss)}, step=step)
            val_outputs = post_trans(logits)



        elif config.finetune.model_choose == "RobustMseg":
            logits, _, _, _ = model(modal_TC[0], modal_VC[0], modal_VG[0])
            log = ''
            total_loss = loss_functions['dice_loss'](logits, gt)
            # accelerator.log({'Val/' + 'dice_loss': float(total_loss)}, step=step)
            val_outputs = post_trans(logits)



        elif config.finetune.model_choose == "MoSID":

            _, _, _, _, pred = model(modal_VG[0], modal_TC[0], modal_VC[0], InforVG, InforTC, InforVC)
            log = ''
            total_loss = loss_functions['dice_loss'](pred, gt)
            # accelerator.log({'Val/' + 'dice_loss': float(total_loss)}, step=step)
            val_outputs = post_trans(pred)



        elif config.finetune.model_choose == "H2Aseg":
            pred, _, _, _, _ = model(modal_VG[0], modal_TC[0], modal_VC[0])
            log = ''
            total_loss = loss_functions['dice_loss'](pred[:, 1:2, :, :], gt)
            # accelerator.log({'Val/' + 'dice_loss': float(total_loss)}, step=step)
            val_outputs = post_trans(pred[:, 1:2, :, :])



        elif config.finetune.model_choose == "MusoMamba" or config.finetune.model_choose == "AttMamba":
            logits = model(modal_TC[0], modal_VC[0], modal_VG[0])
            if config.models.MusoMamba.deep_supervision:
                logits = logits[0]

            total_loss = 0
            log = ''
            # for name in loss_functions:
            focal_loss = loss_functions['focal_loss'](logits, gt)
            dice_loss = loss_functions['dice_loss'](logits, gt)
            # loss = loss_functions[name](logits, gt)
            log += f' focal_loss {float(focal_loss):1.5f} '
            log += f' dice_loss {float(dice_loss):1.5f} '
            total_loss += focal_loss + dice_loss
            val_outputs = post_trans(logits)


        else:
            print("Choose model!")
            assert 0

        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=gt)
        accelerator.print(
            f' Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}',
            flush=True)
        # accelerator.log({'Val/' + 'dice_loss': float(total_loss)}, step=step)


    metric = {}

    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metrics[metric_name].reset()
        metric.update({
            f'val_{metric_name}': float(batch_acc.mean())})

    # accelerator.log(metric, step=epoch)

    return torch.Tensor([metric['val_dice_metric']]).to(accelerator.device), metric


if __name__ == '__main__':

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:1"

    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.model_choose + str(datetime.now())
    kwargs = DDPK(find_unused_parameters=True)



    accelerator = Accelerator(cpu=False)

    accelerator.print(objstr(config))

    include_background = False
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=include_background,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        'miou_metric': monai.metrics.MeanIoU(include_background=include_background),
        'f1': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name='f1 score'),
        'precision': monai.metrics.ConfusionMatrixMetric(include_background=include_background,
                                                         metric_name="precision"),
        'recall': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="recall"),
        'sensitivity': monai.metrics.ConfusionMatrixMetric(include_background=include_background,
                                                           metric_name="sensitivity"),
        'specificity': monai.metrics.ConfusionMatrixMetric(include_background=include_background,
                                                           metric_name="specificity"),
        'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=include_background,
                                                             reduction=monai.utils.MetricReduction.MEAN_BATCH,
                                                             get_not_nans=True),
        'ASD': monai.metrics.SurfaceDistanceMetric(include_background=include_background, symmetric=False,
                                                   distance_metric='euclidean', get_not_nans=False)
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
        'dice_ce_loss': monai.losses.DiceCELoss(lambda_dice=1.0, lambda_ce=1.0, sigmoid=True),
        'l2_loss': nn.MSELoss()
    }

    accelerator.print('Load Dataloader...')

    if config.finetune.model_choose == "MoSID":
        from src.MoSID.step3.dataset.loader import get_kfold_multimodal_dataloader

        train_loaders, val_loaders = get_kfold_multimodal_dataloader(config)
    else:

        from src.dataloader import get_kfold_multimodal_dataloader

        train_loaders, val_loaders = get_kfold_multimodal_dataloader(config)


    # 在交叉验证循环外初始化存储结果的列表
    all_fold_results = []

    for k,  val_loader in enumerate(val_loaders):
        accelerator.print('Load Model...')
        model = give_model(config)


        # best_dice = torch.tensor(0)

        model, val_loader = accelerator.prepare(model, val_loader)
        # best_dice = best_dice.to(accelerator.device)

        # 开始验证
        accelerator.print("Start validation!")

        accelerator.print(f'Load pretrained model from {os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_best/best.pth')
        model.load_state_dict(torch.load(f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_best/best.pth"))

        # 验证
        best_dice, best_results = val_one_epoch(model, loss_functions, val_loader,
                                                            config, metrics,
                                                            post_trans, accelerator)

        accelerator.print(f"best dice: {best_dice}")
        accelerator.print(f"best results : {best_results}")

        # 每折验证完成后，保存该折的最佳指标
        fold_metrics = {
            'fold': k + 1,  # 折数从1开始
        }

        # 合并其他指标
        fold_metrics.update(best_results)
        all_fold_results.append(fold_metrics)
        accelerator.print(f"The {k + 1}-th fold validation is completed. Best Dice: {best_dice.item()}")

    # 在所有折的验证完成后执行
    if accelerator.is_local_main_process:  # 确保只在主进程执行

        # 将结果转换为DataFrame便于处理
        results_df = pd.DataFrame(all_fold_results)

        # 打印所有折的指标值
        print("\nAll Metric:")
        print(results_df.to_string(index=False))  # 不显示索引

        # 提取数值型指标（排除'fold'列）
        metric_columns = [col for col in results_df.columns if col != 'fold']

        # 计算均值和标准差
        metrics_summary = {}
        for col in metric_columns:
            values = results_df[col].values
            metrics_summary[col] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'value': values
            }

        # 打印汇总结果
        print("\nSummary Metric (mean ± std):")
        for metric, stats in metrics_summary.items():
            # 保留4位小数格式化输出
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")