import os
import json
import torch
import pprint
import argparse
import numpy as np
from wilds import *
from tqdm import tqdm
import torchvision.transforms as transforms

# --
_GLOBAL_SEED = 1
np.random.seed(_GLOBAL_SEED)
# --

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file-dir",
    type=str,
    default='subsets/camelyon17_subsets1/',
    help="directory to save low-shot subsets in",
)
parser.add_argument(
    "--data-path",
    type=str,
    default='../datasets/wilds/data/',
    help="path to load WILDS dataset(s) from",
)
parser.add_argument(
    "--dataset",
    type=str,
    default='camelyon17',
    help="WILDS dataset name",
)
parser.add_argument(
    "--subset",
    type=int,
    default=7500,
    help="number of images per class in the subset",
)


def main(args):
    file_dir = args.file_dir
    data_path = args.data_path
    dset_name = args.dataset
    nimgs_class = args.subset

    # -- load the full dataset, set download to true if required
    full_dset = get_dataset(dataset=dset_name, root_dir=data_path, download=False)
    CLASS_COUNTS_PATH = f'src/create_subsets/{dset_name}_class_counts.json'
    class_counts = json.load(open(CLASS_COUNTS_PATH))

    # -- create training subset
    train_data = full_dset.get_subset(
        split = "train", frac = 1, transform = transforms.ToTensor()
    )
    indices = [i for i in range(len(train_data))]

    # -- dump the relative image paths and labels in a text file
    seen_classes = dict()
    file_path = os.path.join(file_dir, f'{nimgs_class}imgs_class.txt')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file = open(file_path, 'w')

    print("\nCreating train subset...")
    for idx in tqdm(indices):
        img_label = train_data[idx][1].item()
        # -- mfw when fixing the relative path bug ✍️🗿
        rel_img_path = train_data.dataset._input_array[train_data.indices[idx]]
        if img_label not in seen_classes:
            seen_classes[img_label] = 1
        elif seen_classes[img_label] < nimgs_class:
            seen_classes[img_label] += 1
        else: continue
        file.write(rel_img_path + "\t" + str(img_label) + "\n")
    print(f"Count of images with exactly {nimgs_class} per class: \n", \
        dict(sorted(seen_classes.items())))

    file.close()
    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    pprint.pprint(args)
    main(args)