
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, mixup_fn=None,
                    log_writer=None,
                    args=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs = torch.cat([samples["input0"], samples["input1"]])
        targets = torch.cat([samples["mask0"], samples["mask1"]])
        instance_ids = torch.cat([samples["instance_id"], samples["instance_id"]])

        if args.mixup:
            batch = mixup_fn({"input": imgs, "instance_id": instance_ids})
            imgs = batch["input"]
            instance_ids = batch["instance_id"]

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).to(torch.float16)
        instance_ids = instance_ids.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():

            mae_loss, _, _, sscd_loss, sscd_stats = model(imgs, targets, instance_ids, mixup=args.mixup, mask_ratio=args.mask_ratio)
        
        
        metric_logger.update(positive_sim=sscd_stats["positive_sim"].item())
        metric_logger.update(negative_sim=sscd_stats["negative_sim"].item())
        metric_logger.update(nearest_negative_sim=sscd_stats["nearest_negative_sim"].item())
        metric_logger.update(InfoNCE=sscd_stats["InfoNCE"].item())
        metric_logger.update(entropy=sscd_stats["entropy"].item())
        metric_logger.update(mae_loss=mae_loss.item())
        metric_logger.update(sscd_loss=sscd_loss.item())

        loss = args.mse_weight*mae_loss + sscd_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            InfoNCE_value = sscd_stats["InfoNCE"].item()
            entropy_value = sscd_stats["entropy"].item()
            mae_value = mae_loss.item()
            print(f"mae_loss: {mae_value} , infonce_loss: {InfoNCE_value} , entropy: {entropy_value} ")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('positive_sim', sscd_stats["positive_sim"].item(), epoch_1000x)
            log_writer.add_scalar('negative_sim', sscd_stats["negative_sim"].item(), epoch_1000x)
            log_writer.add_scalar('nearest_negative_sim', sscd_stats["nearest_negative_sim"].item(), epoch_1000x)
            log_writer.add_scalar('InfoNCE', sscd_stats["InfoNCE"].item(), epoch_1000x)
            log_writer.add_scalar('entropy', sscd_stats["entropy"].item(), epoch_1000x)
            log_writer.add_scalar('mae_loss', mae_loss.item(), epoch_1000x)
            log_writer.add_scalar('sscd_loss', sscd_loss.item(), epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}