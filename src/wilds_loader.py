from PIL import Image
import os, torch, numpy as np
from wilds import get_dataset
from src.transforms import default_transforms
import torch.utils.data as data
from logging import getLogger
logger = getLogger()


def init_data(
    transform=None,
    batch_size=16,
    pin_mem=True,    
    num_workers=2,
    world_size=1,
    rank=0,
    root_path="/srv/datasets/wilds/data",
    image_folder="iwildcam_v2.0",
    training=True,
    val_split="id_val",
    drop_last=False,
    subset_file=None,
    eval_type='lineval',
    model_name='deit_base'
):
    dataset_name = image_folder.split('_')[0]
    full_dset = get_dataset(dataset=dataset_name, root_dir=root_path, download=False)
    transform = default_transforms(image_folder, training, eval_type, model_name) if \
        transform is None else transform
    if training:
        if subset_file:
            # -- Assumes that the training subset is saved offline
            dataset = WILDSSubset(root_path, image_folder, subset_file, transform)
        else:
            dataset = full_dset.get_subset(split="train", frac=1, transform=transform)
    else:
        dataset = full_dset.get_subset(split=val_split, transform=transform)
    logger.info(f"WILDS {dataset_name} dataset created")

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
    logger.info(f"WILDS {dataset_name} data loader created")

    return (data_loader, dist_sampler)


class WILDSSubset(data.Dataset):
    def __init__(self, root, folder, subset_file, transform=None
    ):
        super(WILDSSubset, self).__init__()
        self.root = root
        self.folder = folder
        self.subset_file = subset_file
        self.samples = self.get_samples()
        self.transform = transform

    def get_samples(self):
        samples = []
        with open(self.subset_file, 'r') as file:
            for line in file:
                rel_path, label = line.rstrip().split('\t')
                if 'iwildcam' in self.folder:
                    path = os.path.join(self.root, self.folder, "train", rel_path)
                elif 'camelyon' in self.folder:
                    path = os.path.join(self.root, self.folder, rel_path)
                else:
                    raise Exception(f"Dataset in folder {image_folder} is not supported.")
                samples.append((str(path), int(label)))
        return samples

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.samples)

    def name(self):
        return 'WILDSSubset'
