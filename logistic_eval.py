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
    '--port', type=int,
    default=40111,
    help='port')
parser.add_argument(
    '--lambd', type=float,
    default=0.00025,
    help='regularization')
parser.add_argument(
    '--penalty', type=str,
    help='regularization for logistic classifier',
    default='l2',
    choices=[
        'l2',
        'elastic-net'
    ])
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
    help='whether to standardize images before feeding to network')
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
    lambd,
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
    penalty='l2',
    model_name=None,
    normalize=True,    
    device_str='cuda:0'
):
    global NUM_CLASSES, REMAP_DICT
    device = torch.device(device_str)
    if 'cuda' in device_str:
        torch.cuda.set_device(device)

    # -- Define file names used to save computed embeddings (for efficient
    # -- reuse if running the script more than once)
    subset_tag = '-'.join(subset_path.split('/')).split('.txt')[0] if subset_path is not None \
        else 'imagenet_subsets1-100percent'
    train_embs_path = os.path.join(pretrained, f'train-features-{subset_tag}-{fname}')
    # -- Save embeddings for each test dataset separately
    dataset_tag = '-'.join(image_folder_test.split('/'))
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
        root_path = root_path_train if training else root_path_test
        image_folder = image_folder_train if training else image_folder_test
        subset_file = subset_path if 'wilds' in root_path or training else None
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
        NUM_CLASSES = len(set(sorted(labs.cpu().tolist())))
        logger.info(f'loaded embs of shape {embs.shape}')
    else:
        data_loader = init_pipe(True)
        embs, labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=mask_frac,
            data_loader=data_loader,
            encoder=encoder,
            make_remap_dict=True)
        torch.save({
            'embs': embs,
            'labs': labs
        }, train_embs_path)
        logger.info(f'saved train embs of shape {embs.shape}')
    # -- Normalize embeddings
    cyan.preprocess(embs, normalize=normalize, columns=False, centering=True)

    # -- Fit Logistic Regression Classifier
    if 'camelyon' in image_folder_train:
        classifier = cyan.BinaryClassifier(loss='logistic', penalty=penalty, fit_intercept=False)
    else:
        classifier = cyan.MultiClassifier(loss='multiclass-logistic', penalty=penalty, fit_intercept=False)
    lambd /= len(embs)
    classifier.fit(
        embs.numpy(),
        labs.numpy(),
        it0=10,
        lambd=lambd,
        lambd2=lambd,
        nthreads=-1,
        tol=1e-3,
        solver='auto',
        seed=0,
        max_epochs=300)

    # -- Evaluate and log
    train_score = classifier.score(embs.numpy(), labs.numpy())
    train_top1 = 100. * train_score
    # -- (save train top-1 accuracy)
    logger.info(f'train top-1: {train_top1}')
    # -- (save train per-class accuaracy)
    preds = classifier.predict(embs.numpy())
    conf_mat = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for l, p in zip(labs, preds): conf_mat[max(0, int(l)), max(0, int(p))] += 1
    tot_per_cls, corr_per_cls = conf_mat.sum(axis=1), conf_mat.diagonal()
    per_cls_acc = corr_per_cls[tot_per_cls != 0] / tot_per_cls[tot_per_cls != 0]
    train_avg = 100. * per_cls_acc.mean()
    logger.info(f'train avg: {train_avg}\n\n')

    # -- If test embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    data_loader = init_pipe(False)
    val_projection_fn = getattr(data_loader, 'project_logits', None)
    if preload and os.path.exists(test_embs_path):
        checkpoint = torch.load(test_embs_path, map_location='cpu')
        test_embs, test_labs = checkpoint['embs'], checkpoint['labs']
        if len(REMAP_DICT): 
            test_labs = [REMAP_DICT[test_labs[i]] for i in range(len(test_labs))]
        logger.info(f'loaded test embs of shape {test_embs.shape}')
    else:
        test_embs, test_labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=0.0,
            data_loader=data_loader,
            encoder=encoder,
            make_remap_dict=False)
        torch.save({
            'embs': test_embs,
            'labs': test_labs
        }, test_embs_path)
        logger.info(f'saved test embs of shape {test_embs.shape}')
    # -- Normalize embeddings
    cyan.preprocess(test_embs, normalize=normalize, columns=False, centering=True)

    # -- (save test top1 accuracy)
    test_preds = classifier.predict(test_embs.numpy())
    test_preds[test_preds < 0] = 0

    if val_projection_fn:
        onehot = torch.zeros(test_preds.size, NUM_CLASSES)
        onehot[np.arange(test_preds.size), test_preds] = 1
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


def make_embeddings(
    blocks,
    device,
    mask_frac,
    data_loader,
    encoder,
    make_remap_dict,
    epochs=1,
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

    if make_remap_dict:
        uniq_classes = sorted(list(set(l_mem)))
        REMAP_DICT = {uniq_classes[i]:i for i in range(len(uniq_classes))}
        NUM_CLASSES = len(REMAP_DICT)

    if len(REMAP_DICT):    
        l_mem = [REMAP_DICT[l_mem[i]] for i in range(len(l_mem))]

    l_mem = torch.tensor(l_mem)
    logger.info(z_mem.shape)
    logger.info(l_mem.shape)
    return z_mem, l_mem


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
    args = parser.parse_args()
    pp.pprint(args)
    os.environ["MASTER_PORT"] = str(args.port)
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        logger.addHandler(logging.FileHandler(args.log_file, mode='w'))

    main(
        blocks=1,        
        lambd=args.lambd,
        penalty=args.penalty,
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