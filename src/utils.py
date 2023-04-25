# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import math
import torch

import src.deit as deit
from src.classifier import LinearClassifier, distLinear
import src.resnet50 as resnet

from torch.optim import SGD, Adam

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from logging import getLogger

logger = getLogger()


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


def init_distributed(port=40111, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception:
        world_size, rank = 1, 0
        logger.info('distributed training not available')

    return world_size, rank


class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats


def init_model(
    device,
    num_classes,
    training,
    r_enc_path,
    world_size,
    ref_lr,
    num_epochs,
    its_per_epoch,
    num_blocks=1,
    normalize=True,
    finetuning=False,
    image_folder='imagenet/',
    eval_type='lineval',
    model_name='deit_base',
    warmup_epochs=0,
    weight_decay=0,
    nesterov=True,
    dampening=0.0,
):
    # -- init model and freeze parameters based on finetuning type
    if 'deit' in model_name:
        encoder = deit.__dict__[model_name]()
        emb_dim = 384 if 'small' in model_name else 768 if 'base' in model_name else 1024
        # emb_dim *= num_blocks
        encoder.fc = None
        encoder.norm = None
        if finetuning == 'block':
            for n, p in encoder.named_parameters():
                if 'blocks' not in n:
                    p.requires_grad_(False); continue
                n_blk = int(n.split('.')[1])
                if len(encoder.blocks) - n_blk > num_blocks: p.requires_grad_(False)
                else: p.requires_grad_(True)
        else:
            grad_flag = True if finetuning else False
            for n, p in encoder.named_parameters(): p.requires_grad_(grad_flag)
    elif 'resnet' in model_name:
        encoder = resnet.__dict__[model_name](output_dim=0, eval_mode=False)
        emb_dim = 2048 if model_name == 'resnet50' else 4096
        grad_flag = True if finetuning else False
        for n, p in encoder.named_parameters(): p.requires_grad_(grad_flag)
    elif 'clip' in model_name:
        # NOTE: -- CLIP import can somehow lead to DDP issues
        from clip.model import VisionTransformer, ModifiedResNet
        if 'vitb16' in model_name:
            emb_dim = 512
            encoder = VisionTransformer(input_resolution=224, patch_size=16, \
                width=768, layers=12, heads=12, output_dim=emb_dim)
        elif 'rn50' in model_name:
            emb_dim = 1024
            encoder = ModifiedResNet(input_resolution=224, layers=(3, 4, 6, 3), \
                heads=32, width=64, output_dim=emb_dim)
        grad_flag = True if finetuning else False
        for n, p in encoder.named_parameters(): p.requires_grad_(grad_flag) 
    else:
        raise Exception(f"Model {model_name} is not supported.")
        exit(0)

    # -- different linear classifiers based on eval type
    if eval_type == 'lineval':  
        linear_classifier = LinearClassifier(emb_dim, num_classes, normalize).to(device)
    elif eval_type == 'bslplpl':
        linear_classifier = distLinear(emb_dim, num_classes, normalize).to(device)
    elif eval_type == 'zeroshot':  
        linear_classifier = LinearClassifier(emb_dim, num_classes, True, False).to(device)
    else:
        raise Exception(f"Evaluation type {eval_type} is not supported.")
        exit(0)

    # -- load pretrained weights for encoder and linear classifer if available
    load_pretrained(
        r_path=r_enc_path,
        encoder=encoder,
        linear_classifier=linear_classifier,
        model_name=model_name)
    encoder.to(device)
    linear_classifier.to(device)

    # -- add encoder params to param groups if full-FT  
    param_groups = [
        {'params': (p for n, p in linear_classifier.named_parameters()
                    if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
        {'params': (p for n, p in linear_classifier.named_parameters()
                    if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
         'weight_decay': 0}
    ]
    if finetuning:
        param_groups.append({'params': (p for n, p in encoder.named_parameters()
                    if p.requires_grad and ('fc' not in n))})

    # -- different optimizers and schedulers depending on dataset
    optimizer, scheduler = None, None
    if 'imagenet' in image_folder:
        optimizer = SGD(param_groups, lr=ref_lr, weight_decay=weight_decay, \
            momentum=0.9, dampening=dampening, nesterov=nesterov)
        scheduler = WarmupCosineSchedule(optimizer, warmup_epochs*its_per_epoch, \
            start_lr=ref_lr, ref_lr=ref_lr, T_max=num_epochs*its_per_epoch)
    elif 'iwildcam' in image_folder:
        optimizer = Adam(param_groups, weight_decay=weight_decay, lr=ref_lr)
    elif 'camelyon' in image_folder:
        optimizer = SGD(param_groups, lr=ref_lr, weight_decay=weight_decay, \
            momentum=0.9, dampening=dampening, nesterov=nesterov)
    else:
        raise Exception(f"Dataset in folder {image_folder} is not supported.")
        exit(0)
 
    # -- DDP encapsulation for multi-gpu training and evaluation
    if world_size > 1:
        linear_classifier = DistributedDataParallel(linear_classifier)
        if finetuning:
            encoder = DistributedDataParallel(encoder)

    return encoder, linear_classifier, optimizer, scheduler


def load_pretrained(
    r_path,
    encoder,
    linear_classifier=None,
    model_name='deit_base'
):
    checkpoint = torch.load(r_path, map_location='cpu')
    logger.info(f'model name: {model_name}')
    logger.info(f'checkpoint path: {r_path}')
    enc_checkpoint = checkpoint['target_encoder'] if 'target_encoder' in checkpoint else \
        checkpoint['state_dict'] if 'state_dict' in checkpoint else \
        checkpoint['model'] if 'model' in checkpoint else \
        checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint
    pretrained_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_checkpoint.items()}

    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in encoder and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained encoder with msg: {msg}')

    if linear_classifier is not None:
        lin_checkpoint = checkpoint['linear_classifier'] if 'linear_classifier' in checkpoint else \
            checkpoint['state_dict'] if 'state_dict' in checkpoint else \
            checkpoint['model'] if 'model' in checkpoint else checkpoint
        pretrained_dict = {k.replace('module.', ''): v for k, v in lin_checkpoint.items()}
        try:
            pretrained_dict = {
                'norm.weight': pretrained_dict['head.norm.weight'], 
                'norm.bias': pretrained_dict['head.norm.bias'],
                'linear.weight': pretrained_dict['head.linear.weight'],
                'linear.bias': pretrained_dict['head.linear.bias']
            }
        except Exception:
            pass
        try:
            msg = linear_classifier.load_state_dict(pretrained_dict, strict=True)
            logger.info(f'loaded pretrained linear classifier with msg: {msg}')
        except Exception:
            logger.info(f'failed to load checkpoint, using random init for linear classifier')
            pass

    del checkpoint
    return encoder, linear_classifier
