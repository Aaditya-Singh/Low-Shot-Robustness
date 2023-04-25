# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import sys
import numpy as np

import torch
from src.utils import *
from src.data_manager import init_data as init_inet_data
from src.wilds_loader import init_data as init_wilds_data
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):

    # -- META
    model_name = args['meta']['model_name']
    port = args['meta']['master_port']
    load_checkpoint = args['meta']['load_checkpoint']
    training = args['meta']['training']
    finetuning = args['meta']['finetuning']
    eval_type = args['meta']['eval_type']
    device = torch.device(args['meta']['device'])
    if 'cuda' in args['meta']['device']:
        torch.cuda.set_device(device)

    # -- DATA
    root_path_train = args['data']['root_path_train']
    image_folder_train = args['data']['image_folder_train']
    subset_file = args['data']['subset_file']
    root_path_test = args['data']['root_path_test']
    image_folder_test = args['data']['image_folder_test']
    num_classes = args['data']['num_classes']
    val_split = args['data']['val_split']

    # -- OPTIMIZATION    
    ref_lr = args['optimization']['lr']
    num_epochs = args['optimization']['epochs']
    batch_size = args['optimization']['batch_size']
    num_blocks = args['optimization']['num_blocks']
    l2_normalize = args['optimization']['normalize']
    wd = float(args['optimization']['weight_decay'])
    nesterov = args['optimization']['nesterov']
    dampening = args['optimization']['dampening']

    # -- LOGGING
    folder = args['logging']['folder']
    tboard_folder = args['logging']['tboard_folder']
    tag = args['logging']['write_tag']
    r_file_enc = args['logging']['pretrain_path']

    # -- log/checkpointing paths
    r_enc_path = os.path.join(folder, r_file_enc)
    w_enc_path = os.path.join(folder, f'{tag}.pth.tar')
    tboard_path = os.path.join(tboard_folder, f'{tag}')
    os.makedirs(tboard_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tboard_path)

    # -- init distributed
    world_size, rank = init_distributed(port)
    logger.info(f'initialized rank/world-size: {rank}/{world_size}')

    # -- optimization/evaluation params
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    if not training:
        load_checkpoint = True
        num_epochs = 1

    # -- init loss
    criterion = torch.nn.CrossEntropyLoss()

    # -- use subset file if mentioned in configs
    root_path = root_path_train if training else root_path_test
    init_data = init_wilds_data if 'wilds' in root_path else init_inet_data
    data_loader, dist_sampler = init_data(
        transform=None,
        training=training,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        root_path=root_path_train,
        image_folder=image_folder_train,        
        subset_file=subset_file,
        eval_type=eval_type,
        model_name=model_name)

    ipe = len(data_loader)
    logger.info(f'initialized data-loader (ipe {ipe})')

    # -- make val data transforms and data loaders/samples
    val_data_loader, val_dist_sampler = init_data(
        transform=None,
        batch_size=batch_size,        
        world_size=world_size,
        rank=rank,
        root_path=root_path_test,
        image_folder=image_folder_test, 
        training=False,
        val_split=val_split,     
        drop_last=False,
        subset_file=subset_file,
        eval_type=eval_type,
        model_name=model_name)
    val_projection_fn = getattr(val_data_loader, 'project_logits', None)
    logger.info(f'initialized val data-loader (ipe {len(val_data_loader)})')

    # -- init model and optimizer
    encoder, linear_classifier, optimizer, scheduler = init_model(
        device=device,
        num_classes=num_classes,
        num_blocks=num_blocks,
        normalize=l2_normalize,
        training=training,        
        r_enc_path=r_enc_path,
        its_per_epoch=ipe,
        world_size=world_size,
        ref_lr=ref_lr,
        weight_decay=wd, 
        num_epochs=num_epochs,
        model_name=model_name,
        finetuning=finetuning,
        image_folder=image_folder_train,
        eval_type=eval_type,
        nesterov=nesterov,
        dampening=dampening)
    # logger.info(encoder)

    best_acc = 0
    start_epoch = 0
    if not training:
        logger.info('putting encoder in eval mode')
        encoder.eval()
        logger.info('putting linear classifier in eval mode')
        linear_classifier.eval()

    # -- log number of trainable parameters for sanity check
    encoder_wo_ddp = encoder.module if isinstance(encoder, DistributedDataParallel) \
        else encoder    
    encoder_params = sum(p.numel() for n, p in encoder_wo_ddp.named_parameters()
                        if p.requires_grad and ('fc' not in n))
    logger.info(f"Encoder trainable parameters: {encoder_params}")   
    linear_classifier_params = sum(p.numel() for n, p in linear_classifier.named_parameters()
                        if p.requires_grad)
    logger.info(f"Linear classifier trainable parameters: {linear_classifier_params}")
    total_params = encoder_params + linear_classifier_params
    logger.info(f"Total trainable parameters: {total_params}")

    for epoch in range(start_epoch, num_epochs):

        def train_step():
            # -- update distributed-data-loader epoch
            dist_sampler.set_epoch(epoch); encoder.train()
            top1_correct, avg_acc, total = 0, 0, 0
            conf_mat = torch.zeros(num_classes, num_classes)
            for i, data in enumerate(data_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    # outputs = encoder_wo_ddp.forward_blocks(inputs, num_blocks)
                    outputs = encoder(inputs)
                    outputs = linear_classifier(outputs)
                loss = criterion(outputs, labels)
                total += inputs.shape[0]
                top1_correct += float(outputs.max(dim=1).indices.eq(labels).sum())
                top1_acc = 100. * top1_correct / total
                preds = outputs.max(dim=1).indices.detach().clone()
                for l, p in zip(labels, preds): conf_mat[l, p] += 1
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None: scheduler.step()
                    optimizer.zero_grad()
                if i % log_freq == 0:
                    logger.info('[%d, %5d] %.3f%% (loss: %.3f)'
                                % (epoch + 1, i, top1_acc, loss))
                    # -- add train loss to tboard
                    writer.add_scalars('loss', {'train': loss.item()}, (i + 1)*(epoch + 1))
            top1_acc = 100. * top1_correct / total
            # -- get per-class accuracies from confusion matrix and average
            tot_per_cls, corr_per_cls = conf_mat.sum(axis=1), conf_mat.diagonal()
            per_cls_acc = corr_per_cls[tot_per_cls != 0] / tot_per_cls[tot_per_cls != 0]
            avg_acc = 100. * per_cls_acc.mean()
            return top1_acc, avg_acc

        def val_step():
            encoder.eval()
            top1_correct, avg_acc, total = 0, 0, 0
            conf_mat = torch.zeros(num_classes, num_classes)
            for i, data in enumerate(val_data_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    # outputs = encoder_wo_ddp.forward_blocks(inputs, num_blocks)
                    outputs = encoder(inputs)
                    outputs = linear_classifier(outputs)
                if val_projection_fn:
                    outputs = val_projection_fn(outputs, device)
                total += inputs.shape[0]
                top1_correct += outputs.max(dim=1).indices.eq(labels).sum()
                top1_acc = 100. * top1_correct / total
                preds = outputs.max(dim=1).indices.detach().clone()
                for l, p in zip(labels, preds): conf_mat[l, p] += 1
            top1_acc = AllReduce.apply(top1_acc)
            # -- get per-class accuracies from confusion matrix and average
            tot_per_cls, corr_per_cls = conf_mat.sum(axis=1), conf_mat.diagonal()
            per_cls_acc = corr_per_cls[tot_per_cls != 0] / tot_per_cls[tot_per_cls != 0]
            avg_acc = 100. * per_cls_acc.mean()
            logger.info('[%d, %5d] %.3f%%, %.3f%%' % (epoch + 1, i, top1_acc, avg_acc))
            return top1_acc, avg_acc

        train_top1, train_avg = 0., 0.
        # -- train only if training mode is on in configs
        if training:
            train_top1, train_avg = train_step()
        with torch.no_grad():
            val_top1, val_avg = val_step()

        log_str = 'train top-1:' if training else 'test top-1:'
        logger.info('[%d] (%s %.3f%%) (val top-1: %.3f%%)'
                    % (epoch + 1, log_str, train_top1, val_top1))
        log_str = 'train avg:' if training else 'test avg:'
        logger.info('[%d] (%s %.3f%%) (val avg: %.3f%%)'
                    % (epoch + 1, log_str, train_avg, val_avg))                    
        # -- add train and val per-class accs to tboard
        writer.add_scalars('top_1_accuracy', {'train': train_top1, 'val': val_top1}, \
            epoch + 1)
        writer.add_scalars('per_class_accuracy', {'train': train_avg, 'val': val_avg}, \
            epoch + 1)

        # NOTE: logging/checkpointing with top-1 or per-class accuracy
        curr_acc = val_top1 if 'ImageNet' in image_folder_train else val_avg
        if training and (rank == 0) and (best_acc < curr_acc):
            best_acc = curr_acc
            save_dict = {
                'target_encoder': encoder_wo_ddp.state_dict(),
                'linear_classifier': linear_classifier.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch + 1,
                'world_size': world_size,
                'best_acc': best_acc,
                'batch_size': batch_size,
                'lr': ref_lr,
            }
            torch.save(save_dict, w_enc_path)

    writer.close()
    return train_top1, train_avg, val_top1, val_avg


if __name__ == "__main__":
    main()