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
import torchvision.transforms as transforms

import clip
from src.classifier import LinearClassifier
from src.data_manager import init_data as init_inet_data
from src.wilds_loader import init_data as init_wilds_data

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--port', type=int,
    default=40111,
    help='port')
parser.add_argument(
    '--model-name', type=str,
    help='model architecture')
parser.add_argument(
    "--pretrained", type=str,
    default='pretrained/clip/imagenet_zeroshot_head.pth.tar',
    help='path for zero-shot head weights')
parser.add_argument(
    '--device', type=str,
    default='cuda:0',
    help='device to run script on')
parser.add_argument(
    '--root-path-test', type=str,
    default='/srv/datasets/',
    help='root directory to test data')
parser.add_argument(
    '--image-folder-test', type=str,
    default='ImageNet/',
    help='image directory inside root_path_test')
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

_GLOBAL_SEED, NUM_CLASSES = 0, 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(
    pretrained=None,    
    root_path_test='/srv/share4/datasets/ImageNet/',
    image_folder_test='imagenet/',
    val_split=None,
    model_name='ViT-B/16',
    device_str='cuda:0'
):
    global NUM_CLASSES
    NUM_CLASSES = 2 if 'camelyon' in image_folder_test else 182 if 'iwildcam' in image_folder_test \
        else 1000
    device = torch.device(device_str)
    if 'cuda' in device_str:
        torch.cuda.set_device(device)

    # -- Test dataloader
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))])
    init_data = init_wilds_data if 'wilds' in root_path_test else init_inet_data
    data_loader, _ = init_data(
        transform=transform,
        batch_size=16,
        num_workers=0,
        world_size=1,
        rank=0,
        root_path=root_path_test,
        image_folder=image_folder_test,
        training=False,
        drop_last=False,
        subset_file=None,
        val_split=val_split)
    val_projection_fn = getattr(data_loader, 'project_logits', None)

    # -- Initialize CLIP model and load head weights
    clip_model, _ = clip.load(model_name, device=device)
    emb_dim = 512 if model_name == 'ViT-B/16' else 1024
    linear_classifier = LinearClassifier(emb_dim, NUM_CLASSES, True, False).to(device)
    try:
        checkpoint = torch.load(pretrained)
        pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['linear_classifier'].items()}
        msg = linear_classifier.load_state_dict(pretrained_dict, strict=False)
        logger.info(f'loaded pretrained linear classifier with msg: {msg}')
    except Exception:
        raise Exception("zero-shot head weights need to be saved first.")
        exit(0)

    # -- (save test top1 accuracy)
    test_preds, test_labs = get_predictions(
        device=device,
        data_loader=data_loader,
        clip_model=clip_model,
        linear_classifier=linear_classifier
    )

    if val_projection_fn:
        onehot = torch.zeros(test_preds.shape[0], NUM_CLASSES)
        onehot[np.arange(test_preds.shape[0]), test_preds] = 1
        onehot = val_projection_fn(onehot, 'cpu')
        NUM_CLASSES = onehot.shape[1]
        test_preds = onehot.argmax(axis=1)

    test_top1 = 100. * (np.sum(np.array(test_preds) == np.array(test_labs)) / len(test_preds))
    logger.info(f'test top-1: {test_top1}')

     # -- (save test per-class avg. accuracy)
    conf_mat = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for l, p in zip(test_labs, test_preds): conf_mat[l, int(p)] += 1
    tot_per_cls, corr_per_cls = conf_mat.sum(axis=1), conf_mat.diagonal()
    per_cls_acc = corr_per_cls[tot_per_cls != 0] / tot_per_cls[tot_per_cls != 0]
    test_avg = 100. * per_cls_acc.mean()
    logger.info(f'test avg: {test_avg}\n\n')

    return test_top1, test_avg


def get_predictions(
    device,
    data_loader,
    clip_model,
    linear_classifier,
):
    global NUM_CLASSES
    l_pred, l_corr, ipe = [], [], len(data_loader)

    for itr, data in enumerate(data_loader):
        imgs, corr_lbls = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            img_feats = clip_model.encode_image(imgs).float()
            logit_imgs = linear_classifier(img_feats)
            l_probs = logit_imgs.softmax(dim=-1)
            pred_probs, pred_lbls = torch.max(l_probs, dim=-1)
        pred_lbls, corr_lbls = pred_lbls.cpu().tolist(), corr_lbls.cpu().tolist()
        l_pred.extend(pred_lbls), l_corr.extend(corr_lbls)
        if itr % 50 == 0:
            logger.info(f'[{itr}/{ipe}]')

    l_pred, l_corr = torch.tensor(l_pred), torch.tensor(l_corr)
    print(l_pred.shape); logger.info(l_corr.shape)
    return l_pred, l_corr


if __name__ == '__main__':
    """'main' for launching script using params read from command line"""
    args = parser.parse_args()
    pp.pprint(args)
    os.environ["MASTER_PORT"] = str(args.port)
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        logger.addHandler(logging.FileHandler(args.log_file, mode='w'))

    main(
        pretrained=args.pretrained,
        root_path_test=args.root_path_test,
        image_folder_test=args.image_folder_test,
        val_split=args.val_split,
        model_name=args.model_name,
        device_str=args.device
    )