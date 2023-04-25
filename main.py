# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import torch.multiprocessing as mp

import pprint
import yaml

from src.msn_train import main as msn
from finetune import main as finetune

from src.utils import init_distributed

parser = argparse.ArgumentParser()
# -- add argument for linear evaluation
parser.add_argument(
    '--eval-type', type=str, default='finetune',
    help='whether to run few-shot evaluation or full msn pre-training')
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--log-file', type=str, default=None,
    help='path of file to which write logs to')
#-- distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--port', default=40111, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
    help='url used to set up distributed training')


def process_main(rank, port, eval_type, fname, world_size, devices, log_file):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
    os.environ['NCCL_DEBUG'] = "WARN"  # set to WARN for runtime logging
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    # TODO -- write full length logs
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.addHandler(logging.FileHandler(log_file, mode='w'))
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # -- run pretrain only if eval type is None
    _sel = 'finetune' if eval_type else 'pretrain'
    dump = os.path.join(params['logging']['folder'], f'params-msn-{_sel}.yaml')
    with open(dump, 'w') as f:
        yaml.dump(params, f)

    world_size, rank = init_distributed(port=port, \
        rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    if eval_type == 'pretrain':
        logger.info('Running pre-training')
        return msn(params)
    else:
        logger.info('Running fine-tuning')
        return finetune(params)


if __name__ == '__main__':
    # global args
    args = parser.parse_args()
    pprint.pprint(args)
    num_gpus = len(args.devices)
    mp.spawn(
        process_main,
        nprocs=num_gpus,
        args=(args.port, args.eval_type, args.fname, num_gpus, args.devices, \
            args.log_file))