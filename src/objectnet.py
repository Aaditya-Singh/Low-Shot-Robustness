import os
import PIL
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Sampler
import torchvision.datasets as datasets
from torchvision.transforms import Compose
from src.imagenet_classnames import get_classnames

from logging import getLogger
logger = getLogger()


def init_data(
    transform=None,
    batch_size=128,
    num_workers=6,
    world_size=1,
    rank=0,
    root_path="../datasets/",
    image_folder="objectnet-1.0/images",
    training=False,
    val_split=None,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    pin_mem=True
):
    dataset = ObjectNet(transform, root_path, batch_size, num_workers)
    test_dataset = dataset.get_test_dataset()

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=test_dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers)
    logger.info(f"ObjectNet data loader created")

    return (data_loader, dist_sampler, dataset)


def get_metadata(is_beta):
    if is_beta:
        metadata = Path(__file__).parent / 'objectnet_beta_metadata'
    else:
        metadata = Path(__file__).parent / 'objectnet_metadata'

    with open(metadata / 'folder_to_objectnet_label.json', 'r') as f:
        folder_map = json.load(f)
        folder_map = {v: k for k, v in folder_map.items()}
    with open(metadata / 'objectnet_to_imagenet_1k.json', 'r') as f:
        objectnet_map = json.load(f)

    if is_beta:
        with open(metadata / 'imagenet_to_labels.json', 'r') as f:
            imagenet_map = json.load(f)
            imagenet_map = {v: k for k, v in imagenet_map.items()}
    else:
        with open(metadata / 'pytorch_to_imagenet_2012_id.json', 'r') as f:
            pytorch_map = json.load(f)
            pytorch_map = {v: k for k, v in pytorch_map.items()}

        with open(metadata / 'imagenet_to_label_2012_v2', 'r') as f:
            imagenet_map = {v.strip(): str(pytorch_map[i]) for i, v in enumerate(f)}

    folder_to_ids, class_sublist = {}, []
    classnames = []
    for objectnet_name, imagenet_names in objectnet_map.items():
        imagenet_names = imagenet_names.split('; ')
        imagenet_ids = [int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names]
        class_sublist.extend(imagenet_ids)
        folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

    class_sublist = sorted(class_sublist)
    class_sublist_mask = [(i in class_sublist) for i in range(1000)]
    classname_map = {v: k for k, v in folder_map.items()}
    return class_sublist, class_sublist_mask, folder_to_ids, classname_map


def crop(img):
    width, height = img.size
    cropArea = (2, 2, width - 2, height - 2)
    img = img.crop(cropArea)
    return img


def crop_beta(image, border=2):
    return PIL.ImageOps.crop(image, border=border)


class ImageNet:
    def __init__(self,
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=32,
                num_workers=32,
                classnames='openai',
                distributed=False):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.distributed = distributed

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess,
            )
        sampler = self.get_train_sampler()
        self.sampler = sampler
        kwargs = {'shuffle' : True} if sampler is None else {}
        # print('kwargs is', kwargs)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.distributed else None


    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet'


class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass


def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)


class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform):
        super().__init__(path, transform)

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


class ObjectNetDataset(datasets.ImageFolder):

    def __init__(self, label_map, path, transform):
        self.label_map = label_map
        super().__init__(path, transform=transform)
        self.samples = [
            d for d in self.samples
            if os.path.basename(os.path.dirname(d[0])) in self.label_map
        ]
        self.imgs = self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        label = os.path.basename(os.path.dirname(path))
        return sample, self.label_map[label]


class ObjectNetBase(ImageNet):
    def __init__(self, *args, **kwargs):
        (self._class_sublist,
        self.class_sublist_mask,
        self.folders_to_ids,
        self.classname_map) = get_metadata(self.is_beta())

        super().__init__(*args, **kwargs)

        self.classnames = sorted(list(self.folders_to_ids.keys()))
        self.rev_class_idx_map = {}
        self.class_idx_map = {}
        for idx, name in enumerate(self.classnames):
            self.rev_class_idx_map[idx] = self.folders_to_ids[name]
            for imagenet_idx in self.rev_class_idx_map[idx]:
                self.class_idx_map[imagenet_idx] = idx

        if self.is_beta():
            self.crop = crop_beta
        else:
            self.crop = crop
        self.preprocess = Compose([crop, self.preprocess])
        self.classnames = [self.classname_map[c].lower() for c in self.classnames]

    def is_beta(self):
        raise NotImplementedError

    def populate_train(self):
        pass

    def get_test_dataset(self, subdir='objectnet-1.0/images'):
        valdir = os.path.join(self.location, subdir)
        label_map = {name: idx for idx, name in enumerate(sorted(list(self.folders_to_ids.keys())))}
        return ObjectNetDataset(label_map, valdir, transform=self.preprocess)

    def project_logits(self, logits, device):
        if isinstance(logits, list) or isinstance(logits, tuple):
            return [self.project_logits(l, device) for l in logits]
        if logits.shape[1] == 113:
            return logits
        if torch.is_tensor(logits):
            logits = logits.cpu().numpy()
        logits_projected = np.zeros((logits.shape[0], 113))
        for k, v in self.rev_class_idx_map.items():
            logits_projected[:, k] = np.max(logits[:, v], axis=1).squeeze()
        return torch.tensor(logits_projected).to(device)

    def scatter_weights(self, weights):
        if weights.size(1) == 1000:
            return weights
        new_weights = torch.ones((weights.size(0), 1000)).to(weights.device) * -10e8
        for k, v in self.rev_class_idx_map.items():
            for vv in v:
                new_weights[:, vv] = weights[:, k]
        return new_weights


def accuracy(logits, targets, img_paths, args):
    assert logits.shape[1] == 113
    preds = logits.argmax(dim=1)
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return np.sum(preds == targets), len(preds)


class ObjectNetBetaValClassesBase(ObjectNetBase):

    def get_test_sampler(self):
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self._class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def project_labels(self, labels, device):
        projected_labels =  [self.class_idx_map[int(label)] for label in labels]
        return torch.LongTensor(projected_labels).to(device)


class ObjectNetBetaValClasses(ObjectNetBetaValClassesBase):

    def is_beta(self):
        return True


class ObjectNetValClasses(ObjectNetBetaValClassesBase):

    def is_beta(self):
        return False


class ObjectNet(ObjectNetBase):

    def accuracy(self, logits, targets, img_paths, args):
        return accuracy(logits, targets, img_paths, args)

    def is_beta(self):
        return False


class ObjectNetBeta(ObjectNetBase):

    def accuracy(self, logits, targets, img_paths, args):
        return accuracy(logits, targets, img_paths, args)

    def is_beta(self):
        return True