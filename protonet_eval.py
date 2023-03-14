# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os, sys
import argparse
import logging
import pprint

import torch
import numpy as np
import torch.nn as nn
import cyanure as cyan
import torchvision.transforms as transforms

import src.deit as deit
import src.resnet50 as resnet_models
from clip.model import VisionTransformer, ModifiedResNet

from src.utils import load_pretrained
from src.data_manager import init_data as init_inet_data
from src.wilds_loader import init_data as init_wilds_data

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mask', type=float,
    default=0.0,
    help='regularization')
parser.add_argument(
    '--preload', action='store_true',
    help='whether to preload embs if possible')
parser.add_argument(
    '--fname', type=str,
    help='model architecture')
parser.add_argument(
    '--model-name', type=str,
    help='model architecture')
parser.add_argument(
    '--pretrained', type=str,
    help='path to pretrained model',
    default='')
parser.add_argument(
    '--device', type=str,
    default='cuda:0',
    help='device to run script on')
parser.add_argument(
    '--normalize', type=bool,
    default=True,
    help='whether to standardize images before feeding to nework')
parser.add_argument(
    '--root-path-train', type=str,
    default='/srv/datasets/',
    help='root directory to training data')
parser.add_argument(
    '--image-folder-train', type=str,
    default='ImageNet/',
    help='image directory inside root_path_train')
parser.add_argument(
    '--root-path-test', type=str,
    default='/srv/datasets/',
    help='root directory to test data')
parser.add_argument(
    '--image-folder-test', type=str,
    default='ImageNet/',
    help='image directory inside root_path_test')
parser.add_argument(
    '--subset-path', type=str,
    default=None,
    help='name of dataset to train on')
parser.add_argument(
    '--val-split', type=str,
    default=None,
    help='name of split to evaluate on')    
parser.add_argument(
    '--log-file', type=str,
    default=None,
    help='path of file to which write logs to')

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED, NUM_CLASSES, REMAP_DICT = 0, 0, {}
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(
    blocks,
    mask_frac,
    preload,
    pretrained,
    fname,
    subset_path,
    root_path_train,
    image_folder_train,
    root_path_test,
    image_folder_test,
    val_split=None,
    model_name=None,
    normalize=True,
    device_str='cuda:0'
):
    device = torch.device(device_str)
    if 'cuda' in device_str:
        torch.cuda.set_device(device)

    # -- Define file names used to save computed embeddings (for efficient
    # -- reuse if running the script more than once)
    subset_tag = '-'.join(subset_path.split('/')).split('.txt')[0] if subset_path is not None \
        else 'imagenet_subsets1-100percent'
    train_embs_path = os.path.join(pretrained, f'train-features-{subset_tag}-{fname}')
    # -- Save embeddings for each test dataset separately
    dataset_tag = '-'.join(image_folder_test.split('/'))[:-1]
    test_embs_path = os.path.join(pretrained, f'val-features-{dataset_tag}-{fname}')
    logger.info(train_embs_path)
    logger.info(test_embs_path)

    pretrained = os.path.join(pretrained, fname)

    # -- Function to make train/test dataloader
    def init_pipe(training):
        # -- make data transforms
        transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))])
        # -- init data-loaders/samplers
        subset_file = subset_path if training else None
        root_path = root_path_train if training else root_path_test
        image_folder = image_folder_train if training else image_folder_test
        init_data = init_wilds_data if 'wilds' in root_path else init_inet_data      
        data_loader, _ = init_data(
            transform=transform,
            batch_size=16,
            num_workers=0,
            world_size=1,
            rank=0,
            root_path=root_path,
            image_folder=image_folder,
            training=training,
            drop_last=False,
            subset_file=subset_file,
            val_split=val_split)
        return data_loader

    # -- Initialize the model
    encoder = init_model(
        device=device,
        pretrained=pretrained,
        model_name=model_name)
    encoder.eval()

    # -- If train embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(train_embs_path):
        checkpoint = torch.load(train_embs_path, map_location='cpu')
        embs, labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded embs of shape {embs.shape}')
    else:
        data_loader = init_pipe(True)
        embs, labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=mask_frac,
            data_loader=data_loader,
            encoder=encoder)
        torch.save({
            'embs': embs,
            'labs': labs
        }, train_embs_path)
        logger.info(f'saved train embs of shape {embs.shape}')
    # -- Normalize embeddings
    cyan.preprocess(embs, normalize=normalize, columns=False, centering=True)

    # -- Get cluster embeddings and labels
    clr_embs, clr_labs = make_cluster_embs(embs, labs)
    # -- Evaluate and log
    train_top1, train_avg = calc_accs(embs, labs, clr_embs, clr_labs, None)
    # -- (save train top-1 and per-class avg. accs)
    logger.info(f'train top1: {train_top1}')
    logger.info(f'train avg: {train_avg}')

    # -- If test embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(test_embs_path):
        checkpoint = torch.load(test_embs_path, map_location='cpu')
        test_embs, test_labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded test embs of shape {test_embs.shape}')
    else:
        data_loader = init_pipe(False)
        test_embs, test_labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=0.0,
            data_loader=data_loader,
            encoder=encoder)
        torch.save({
            'embs': test_embs,
            'labs': test_labs
        }, test_embs_path)
        logger.info(f'saved test embs of shape {test_embs.shape}')
    # -- Normalize embeddings
    cyan.preprocess(test_embs, normalize=normalize, columns=False, centering=True)

    # -- Evaluate and log
    try:
        val_projection_fn = getattr(data_loader, 'project_logits', None)
    except Exception:
        val_projection_fn = None
    test_top1, test_avg = calc_accs(test_embs, test_labs, clr_embs, clr_labs, val_projection_fn)
    # -- (save test top-1 and per-class avg. acc)
    logger.info(f'test top1: {test_top1}')
    logger.info(f'test avg: {test_avg}\n\n')

    return test_top1, test_avg


def make_embeddings(
    blocks,
    device,
    mask_frac,
    data_loader,
    encoder,
    epochs=1
):
    ipe = len(data_loader)

    z_mem, l_mem = [], []

    for _ in range(epochs):
        for itr, data in enumerate(data_loader):
            imgs, labels = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                z = encoder(imgs).cpu()
            labels = labels.cpu().tolist()
            z_mem.append(z)
            l_mem.extend(labels)
            if itr % 50 == 0:
                logger.info(f'[{itr}/{ipe}]')

    z_mem = torch.cat(z_mem, 0)
    # NOTE: potentailly remap labels because cyanure can't handle empty classes
    global NUM_CLASSES, REMAP_DICT
    if not len(REMAP_DICT):
        uniq_classes = sorted(list(set(l_mem)))
        REMAP_DICT = {uniq_classes[i]:i for i in range(len(uniq_classes))}
        NUM_CLASSES = len(REMAP_DICT)
    logger.info(f'No. of classes: {NUM_CLASSES}')
    l_mem = [REMAP_DICT[l_mem[i]] for i in range(len(l_mem))]
    l_mem = torch.tensor(l_mem)
    logger.info(z_mem.shape)
    logger.info(l_mem.shape)
    return z_mem, l_mem


# -- Average embeddings for cluster prototypes
def make_cluster_embs(
    embs,
    labs,
    num_classes=1000
):
    n, embs_dim = embs.shape
    clr_embs = torch.zeros(num_classes, embs_dim)
    lab_cnts = torch.zeros(num_classes)
    for i in range(n):
        lab = labs[i]
        lab_cnts[lab] += 1
        clr_embs[lab] += embs[i]    
    # -- Remove fully zero prototypes
    clr_labs = torch.unique(torch.nonzero(clr_embs, as_tuple=True)[0])
    lab_cnts = lab_cnts[clr_labs]
    clr_embs = clr_embs[clr_labs]/lab_cnts.unsqueeze(-1)
    
    logger.info(clr_embs.shape)
    logger.info(clr_labs.shape)

    return clr_embs, clr_labs


# -- Calculate top-1 and per-class avf acc based on L2-distance from prototypes
def calc_accs(embs, labs, clr_embs, clr_labs, val_projection_fn=None):
    global NUM_CLASSES
    l2_dist = torch.cdist(embs, clr_embs, p=2.0)
    _, min_idx = torch.min(l2_dist, dim=-1)
    pred_labs = clr_labs[min_idx]

    if val_projection_fn:
        onehot = torch.zeros(pred_labs.shape[0], NUM_CLASSES)
        onehot[np.arange(pred_labs.shape[0]), pred_labs] = 1
        onehot = val_projection_fn(onehot, 'cpu')
        NUM_CLASSES = onehot.shape[1]
        pred_labs = onehot.argmax(axis=1)

    correct = torch.sum(torch.eq(pred_labs, labs))
    top1_acc = 100. * correct / labs.shape[0]

    conf_mat = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for l, p in zip(labs, pred_labs): conf_mat[l, p] += 1
    tot_per_cls, corr_per_cls = conf_mat.sum(axis=1), conf_mat.diagonal()
    per_cls_acc = corr_per_cls[tot_per_cls != 0] / tot_per_cls[tot_per_cls != 0]
    avg_acc = 100. * per_cls_acc.mean()

    return top1_acc, avg_acc


def init_model(
    device,
    pretrained,
    model_name,
):
    if 'deit' in model_name:
        encoder = deit.__dict__[model_name]()
        encoder.fc = None
        encoder.norm = None
    elif 'resnet' in model_name:
        encoder = resnet_models.__dict__[model_name](output_dim=0, eval_mode=False)
    elif 'clip' in model_name:
        if 'vitb16' in model_name:
            encoder = VisionTransformer(input_resolution=224, patch_size=16, \
                width=768, layers=12, heads=12, output_dim=512)
        elif 'rn50' in model_name:
            encoder = ModifiedResNet(input_resolution=224, layers=(3, 4, 6, 3), \
                heads=32, width=64, output_dim=1024)
    else:
        raise Exception(f"Model {model_name} is not supported.")
        exit(0)

    encoder.to(device)
    encoder, _ = load_pretrained(r_path=pretrained, encoder=encoder, model_name=model_name)

    return encoder


if __name__ == '__main__':
    """'main' for launching script using params read from command line"""
    global args
    args = parser.parse_args()
    pp.pprint(args)
    # TODO -- write full length logs
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        logger.addHandler(logging.FileHandler(args.log_file, mode='w'))
    main(
        blocks=1,
        mask_frac=args.mask,
        preload=args.preload,
        pretrained=args.pretrained,
        fname=args.fname,
        subset_path=args.subset_path,
        root_path_train=args.root_path_train,
        image_folder_train=args.image_folder_train,
        root_path_test=args.root_path_test,
        image_folder_test=args.image_folder_test,
        val_split=args.val_split,     
        model_name=args.model_name,
        normalize=args.normalize,
        device_str=args.device
    )