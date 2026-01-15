import torch
import os
# torch.autograd.set_detect_anomaly(True)
# 安全获取LOCAL_RANK，避免环境变量未设置时报错
# local_rank = int(os.environ["LOCAL_RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 默认为0（单卡场景兼容）
torch.cuda.set_device(local_rank)  # 绑定当前进程到指定GPU
import sys

import numpy
from datetime import datetime
from typing import Dict
import numpy as np
import time
from safetensors.torch import load_model
import torch.nn as nn
import pandas as pd
import monai
import pytz

from src.dataloader import get_kfold_multimodal_dataloader

import yaml
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate.tracking import GeneralTracker
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.models import give_model
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model, MyCustomTracker
from src.loss import KL_divergence


import warnings
warnings.filterwarnings('ignore')

def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                    post_trans: monai.transforms.Compose, accelerator: Accelerator, tracker:GeneralTracker, epoch: int, step: int):
    # 训练
    model.train()
    for i, image_batch in enumerate(train_loader):

        if config.finetune.model_choose == "MoSID":
            modal_TC, modal_VC, modal_VG, recon_TC_VC, recon_TC, recon_VC, all_true = image_batch

            InforVG = recon_TC_VC[0]
            InforTC = torch.abs(recon_TC[0] - all_true[0])
            InforVC = torch.abs(recon_VC[0] - all_true[0])
        else:
            modal_TC, modal_VC, modal_VG = image_batch

        for m in config.dataset.Breast_US.modal_mask:
            if m == "TC":
                modal_TC[0] = torch.ones_like(modal_TC[0])
            if m == "VC":
                modal_VC[0] = torch.ones_like(modal_VC[0])
            if m == "VG":
                modal_VG[0] = torch.ones_like(modal_VG[0])


        gt = modal_VG[1]
        total_loss = 0
        log = ''

        if config.finetune.model_choose == "mmFormer" or config.finetune.model_choose == "RFNet":

            fuse_pred, (TC_pred, VC_pred, VG_pred), preds = model(modal_TC[0], modal_VC[0], modal_VG[0], is_training=True)

            loss_encoder = loss_functions['dice_loss'](TC_pred, gt) + loss_functions['dice_loss'](VC_pred, gt) + loss_functions['dice_loss'](VG_pred, gt)
            loss_decoder = 0
            for j in range(len(preds)):
                loss_decoder += loss_functions['dice_loss'](preds[j],gt)
            loss_output = loss_functions['dice_loss'](fuse_pred,gt)

            total_loss = loss_encoder + loss_decoder + loss_output

            val_outputs = post_trans(fuse_pred)

        elif config.finetune.model_choose == "NestedFormer" or config.finetune.model_choose == "MMEFUNet" \
                or config.finetune.model_choose == "MMCANET" or config.finetune.model_choose == "ASANet" or config.finetune.model_choose == "MMFFNet":
            logits = model(modal_TC[0], modal_VC[0], modal_VG[0])
            total_loss = loss_functions['dice_loss'](logits, gt)

            val_outputs = post_trans(logits)


        elif config.finetune.model_choose == "MAML":
            logits = model(modal_TC[0], modal_VC[0], modal_VG[0], is_training=True)

            max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


            gt2 = max_pooling_2d(gt)

            gt3 = max_pooling_2d(gt2)
            loss_intra = 0
            loss_joint = 0
            for j, output in enumerate(logits):
                if j == 0:
                    loss_joint = loss_functions['dice_ce_loss'](output, gt)

                elif j == 1 or j == 4 or j == 7:
                    loss_intra +=  loss_functions['dice_ce_loss'](output, gt)
                elif j == 2 or j == 5 or j == 8:
                    loss_intra += loss_functions['dice_ce_loss'](output, gt2)

                elif j == 3 or j == 6 or j == 9:
                    loss_intra += loss_functions['dice_ce_loss'](output, gt3)
                else:
                    assert 0
            total_loss = 0.5 * loss_intra + loss_joint

            val_outputs = post_trans(logits[0])


        elif config.finetune.model_choose == "RobustMseg":
            seg_out, recon_out, mu_list, sigma_list = model(modal_TC[0], modal_VC[0], modal_VG[0])

            alpha = 0.1
            dice_ce_loss = loss_functions['dice_ce_loss'](seg_out, gt)
            recon = loss_functions['l2_loss'](recon_out, torch.cat((modal_TC[0], modal_VC[0], modal_VG[0]), dim=1))
            kl = 0.0
            for m in range(3):  # modality
                kl += KL_divergence(mu_list[m], torch.log(torch.square(sigma_list[m])))
            total_loss = dice_ce_loss + alpha*recon*4 + alpha*kl

            val_outputs = post_trans(seg_out)

        elif config.finetune.model_choose == "MoSID":
            max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            out4, out3, out2, out, pred = model(modal_VG[0], modal_TC[0], modal_VC[0], InforVG, InforTC, InforVC)

            gt2 = max_pooling_2d(gt).detach()
            gt3 = max_pooling_2d(gt2).detach()
            gt4 = max_pooling_2d(gt3).detach()

            dice_ce_loss = loss_functions['dice_ce_loss'](pred, gt)
            supervised_loss = (loss_functions['dice_loss'](pred, gt) + loss_functions['dice_loss'](out4, gt4) + loss_functions['dice_loss'](out3, gt3) +
                               loss_functions['dice_loss'](out2, gt2) + loss_functions['dice_loss'](out, gt))
            total_loss = dice_ce_loss + supervised_loss

            val_outputs = post_trans(pred)

        elif config.finetune.model_choose == "H2Aseg":
            pred, out4, out3, out2, out1 = model(modal_VG[0], modal_TC[0], modal_VC[0])
            dice_ce_loss = loss_functions['dice_ce_loss'](pred[:, 1:2, :, :], gt)
            supervised_loss = (loss_functions['dice_loss'](pred[:, 1:2, :, :], gt) + loss_functions['dice_loss'](out4[:, 1:2, :, :], gt) + loss_functions['dice_loss'](out3[:, 1:2, :, :], gt) +
                               loss_functions['dice_loss'](out2[:, 1:2, :, :], gt) + loss_functions['dice_loss'](out1[:, 1:2, :, :], gt))
            total_loss = dice_ce_loss + supervised_loss
            val_outputs = post_trans(pred[:, 1:2, :, :])

        elif config.finetune.model_choose == "MusoMamba" or config.finetune.model_choose == "AttMamba":

            logits = model(modal_TC[0], modal_VC[0], modal_VG[0])

            if config.models.MusoMamba.deep_supervision:
                max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                gts = []
                gt2 = max_pooling_2d(gt)
                gt3 = max_pooling_2d(gt2)
                gt4 = max_pooling_2d(gt3)
                gts.append(gt)
                gts.append(gt2)
                gts.append(gt3)
                gts.append(gt4)

                for j in range(len(gts)):
                    # for name in loss_functions:
                    #     alpth = 1
                    focal_loss = loss_functions['focal_loss'](logits[j], gts[j])
                    dice_loss =  loss_functions['dice_loss'](logits[j], gts[j])
                        # accelerator.log({'Train/' + name: float(loss)}, step=step)
                        # total_loss += alpth * loss
                    total_loss +=  focal_loss + dice_loss
                val_outputs = post_trans(logits[0])

            else:
                focal_loss = loss_functions['focal_loss'](logits, gt)
                dice_loss = loss_functions['dice_loss'](logits, gt)
                total_loss = focal_loss + dice_loss
                val_outputs = post_trans(logits)



        else:
            print("Choose model!")
            assert 0

        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=gt)


        accelerator.backward(total_loss)


        optimizer.step()
        optimizer.zero_grad()

        tracker.add_scalar(tag='Train/Total Loss',
                           scalar_value=float(total_loss),
                           global_step=step)


        # accelerator.log({'Train/Total Loss': float(total_loss),}, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss: {total_loss:1.5f} {log}',
            flush=True)
        step += 1
        # break
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metrics[metric_name].reset()
        metric.update({
        f'train_{metric_name}': float(batch_acc.mean())})
        
    accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}')
    accelerator.log(metric, step=epoch)
    return step


@torch.no_grad()
def val_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                  val_loader: torch.utils.data.DataLoader,
                  config: EasyDict, metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator, tracker:GeneralTracker, epoch: int):
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


        elif config.finetune.model_choose == "NestedFormer" or config.finetune.model_choose == "MMEFUNet" \
                or config.finetune.model_choose == "MMCANET" or config.finetune.model_choose == "ASANet" or config.finetune.model_choose == "MMFFNet":
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
            dice_loss =  loss_functions['dice_loss'](logits, gt)
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
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}',
            flush=True)

        tracker.add_scalar(tag='val/Total Loss',
                           scalar_value=float(total_loss),
                           global_step=step)
        # accelerator.log({'Val/' + 'dice_loss': float(total_loss)}, step=step)
        step += 1

    metric = {}

    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metrics[metric_name].reset()
        metric.update({
        f'val_{metric_name}': float(batch_acc.mean())})
            
    accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}')
    # accelerator.log(metric, step=epoch)

    return torch.Tensor([metric['val_dice_metric']]).to(accelerator.device), metric, step


if __name__ == '__main__':

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:1"

    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.model_choose + str(config.dataset.Breast_US.modal_mask) +str(datetime.now())
    kwargs = DDPK(find_unused_parameters=True)

    tracker = MyCustomTracker(run_name=config.finetune.model_choose, logging_dir=logging_dir)

    accelerator = Accelerator(cpu=False, log_with=tracker, kwargs_handlers=[kwargs])
    Logger(logging_dir if accelerator.is_local_main_process else None)

    # accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir, kwargs_handlers=[kwargs])
    # accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])

    accelerator.print(objstr(config))

    include_background = False
    # inference = monai.inferers.SlidingWindowInferer(roi_size=ensure_tuple_rep(image_size, 2), overlap=0.5,
    #                                                 sw_device=accelerator.device, device=accelerator.device)
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
        train_loaders, val_loaders = get_kfold_multimodal_dataloader(config)

    all_fold_results = []

    for k, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):

        accelerator.print('Load Model...')
        model = give_model(config)

        # 定义训练参数
        optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                      weight_decay=config.trainer.weight_decay,
                                                      lr=config.trainer.lr, betas=(0.9, 0.95))
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                                  max_epochs=config.trainer.num_epochs)

        step = 0
        best_eopch = -1
        val_step = 0
        starting_epoch = 0
        best_dice = torch.tensor(0)
        best_results = []


        model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)
        best_dice = best_dice.to(accelerator.device)

        # 开始训练
        accelerator.print(f"Start fold {k + 1} training!")

        for epoch in range(starting_epoch, config.trainer.num_epochs):

            # 训练
            # start_time = time.time()
            step = train_one_epoch(model, loss_functions, train_loader,
                                   optimizer, scheduler, metrics,
                                   post_trans, accelerator, tracker, epoch, step)
            # end_time = time.time()
            # print("run time:", end_time - start_time)
            # 验证
            mean_dice, metric_results, val_step = val_one_epoch(model, loss_functions, val_loader,
                                                          config, metrics, val_step,
                                                          post_trans, accelerator, tracker, epoch)

            # 保存模型
            if mean_dice > best_dice:
                accelerator.print(f"Saving best model to {os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_best")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                if accelerator.is_main_process:
                    if not os.path.exists(f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_best"):
                        os.makedirs(f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_best")

                    accelerator.save(unwrapped_model.state_dict(),
                                     f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_best/best.pth")

                # accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_best")

                # load_model(model,
                #            f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_best/model.safetensors")


                best_dice = mean_dice
                best_results = metric_results
                best_eopch = epoch
            accelerator.print('Saving Checkpoint...')




            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            if accelerator.is_main_process:
                if not os.path.exists(
                        f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_checkpoint"):
                    os.makedirs(f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_checkpoint")
                accelerator.save(unwrapped_model.state_dict(),
                                 f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_checkpoint/checkpoint.pth")

                # accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_checkpoint")
                # torch.save({'epoch': epoch, 'best_dice': best_dice, 'metric_results': metric_results},
                #            f'{os.getcwd()}/model_store/{config.finetune.model_choose}/fold_{k + 1}_checkpoint/epoch.pth')

                accelerator.print(
                    f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] best_eopch:{best_eopch + 1}, best dice:{best_dice}, Now : mean dice: {mean_dice}, metric_results: {metric_results}')

        accelerator.print(f"best dice: {best_dice}")
        accelerator.print(f"best results : {best_results}")
        accelerator.print(f"best epochs: {best_eopch}")


        # 每折训练完成后，保存该折的最佳指标
        fold_metrics = {
            'fold': k + 1,  # 折数从1开始
            'best_epoch': best_eopch,
        }

        # 合并其他指标
        fold_metrics.update(best_results)
        all_fold_results.append(fold_metrics)
        accelerator.print(f"The {k + 1}-th fold training is completed. Best Dice: {best_dice.item()}")

    #必须完成k折训练后执行
    accelerator.end_training()



    if accelerator.is_main_process:
    # 生成带时间戳的文件名，避免覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_results_path = os.path.join(logging_dir, f"{config.finetune.model_choose}_kfold_results_{timestamp}.csv")
        summary_path = os.path.join(logging_dir, f"{config.finetune.model_choose}_summary_{timestamp}.csv")

        # 1. 保存每折的详细结果
        df_folds = pd.DataFrame(all_fold_results)
        df_folds.to_csv(fold_results_path, index=False)
        accelerator.print(f"The results of each fold have been saved to: {fold_results_path}")

        # 2. 计算并保存汇总统计（平均值和标准差）
        summary = {}
        # 遍历所有指标（排除'fold'和'best_epoch'）
        metric_columns = [col for col in df_folds.columns if col not in ['fold', 'best_epoch']]

        for metric in metric_columns:
            values = df_folds[metric].values
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)

        # 转换为DataFrame并保存
        df_summary = pd.DataFrame([summary])
        df_summary.to_csv(summary_path, index=False)
        accelerator.print(f"Cross validation summary results have been saved to: {summary_path}")