# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from logging import getLogger

import torch
import torchvision
from src.transforms import default_transforms
from src.objectnet import ObjectNet
from src.imagenetv2 import ImageNetV2
from src.imagenet_r import ImageNetR
from src.imagenet_a import ImageNetA

import json
IN1K_CLASS_DICT_PATH = './src/in1k_class_to_idx.json'
in1k_dict = json.load(open(IN1K_CLASS_DICT_PATH))
logger = getLogger()


def init_data(
    transform=None,
    batch_size=128,
    pin_mem=True,
    num_workers=6,
    world_size=1,
    rank=0,
    root_path="/srv/share4/datasets/",
    image_folder="ImageNet/",
    training=True,
    val_split=None,
    drop_last=True,
    subset_file=None,
    eval_type='lineval',
    model_name='deit_base'
):

    transform = default_transforms(image_folder, training, eval_type, model_name) if \
        transform is None else transform

    projection_fn = None
    if 'imagenetv2' in image_folder:
        imagenetv2 = ImageNetV2(transform, root_path)
        dataset = imagenetv2.get_test_dataset(image_folder)
    elif 'objectnet' in image_folder:
        objectnet = ObjectNet(transform, root_path)
        dataset = objectnet.get_test_dataset(image_folder)
        projection_fn = getattr(objectnet, 'project_logits', None)
    elif 'imagenet-a' in image_folder:
        imagenet_a = ImageNetA(transform, root_path)
        dataset = imagenet_a.get_test_dataset()
        projection_fn = getattr(imagenet_a, 'project_logits', None)
    elif 'imagenet-r' in image_folder:
        imagenet_r = ImageNetR(transform, root_path)
        dataset = imagenet_r.get_test_dataset()
        projection_fn = getattr(imagenet_r, 'project_logits', None)
    else:
        dataset = ImageNet(root=root_path, image_folder=image_folder, \
            transform=transform, train=training)
        if subset_file is not None and training:
            dataset = ImageNetSubset(dataset, subset_file)

    logger.info('ImageNet dataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers)
    data_loader.project_logits = projection_fn
    logger.info('ImageNet unsupervised data loader created')

    return (data_loader, dist_sampler)


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root='/srv/datasets/',
        image_folder='ImageNet/',
        transform=None,
        train=True
    ):
        """
        ImageNet

        Dataset wrapper

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        """

        # -- don't use suffix when dataset is imagenet-a/c/r/s
        in_variants = {'-a', '-c', '-r', '-s'}
        ood = any(var in image_folder for var in in_variants)
        suffix = '' if ood else 'train/' if train else 'val/'
        data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')
        transform = default_transforms(image_folder, train) if transform is None else transform
        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')

    # -- use in1k class to idx mapping for in1k and variants
    def __getitem__(self, index):
        path, _ = self.samples[index]
        cls = path.split('/')[-2]
        global in1k_dict
        target = in1k_dict[cls]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target