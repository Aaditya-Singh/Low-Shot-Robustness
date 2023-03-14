from tqdm import tqdm
from wilds import *
import os, json, torch, numpy as np
import torchvision.transforms as transforms

# Load the full dataset, and download it if necessary
dset_name = "camelyon17"
data_path = "/srv/share4/datasets/wilds/data"
full_dset = get_dataset(dataset=dset_name, root_dir=data_path, download=False)

# NOTE: Assumes class to count dict in train set is saved offline
CLASS_COUNTS_PATH = f'src/{dset_name}_class_counts.json'
TEST_SPLITS = []    # id_val, val, id_test, test
class_counts = json.load(open(CLASS_COUNTS_PATH))

# Set random seed for reproducibility
n_subset, ss_ratio, class_bound = 1, 1, 1500
nimgs_class = class_bound // ss_ratio
np.random.seed(n_subset)

################ Create training subset ################
train_data = full_dset.get_subset(
    split = "train", frac = 1, transform = transforms.ToTensor()
)

# Dump the relative image paths and labels in a text file
seen_classes = dict()
indices = [i for i in range(len(train_data))] # np.random.shuffle(indices)
file_path = f"subsets/{dset_name}_subsets{n_subset}/{nimgs_class}imgs_class_train.txt"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
file = open(file_path, 'w')

print("\nCreating train subset")
for idx in tqdm(indices):
    img_label = train_data[idx][1].item()
    # > mfw when fixing the relative path bug ✍️🗿
    rel_img_path = train_data.dataset._input_array[train_data.indices[idx]]
    if class_counts[str(img_label)] < class_bound: continue
    elif img_label not in seen_classes:
        seen_classes[img_label] = 1
    elif seen_classes[img_label] < nimgs_class:
        seen_classes[img_label] += 1
    else: continue
    file.write(rel_img_path + "\t" + str(img_label) + "\n")

print(f"\nImages obtained from classes with at least {class_bound} number of images")
print(f"Count of such images with exactly {nimgs_class} per class: \n", \
    dict(sorted(seen_classes.items())))
file.close()

################ Create test subsets ################
for test_split in TEST_SPLITS:    
    test_data = full_dset.get_subset(
        split = test_split, transform = transforms.ToTensor()
    )

    # Dump the relative image paths and labels in a text file
    indices = [i for i in range(len(test_data))] # np.random.shuffle(indices)
    file_path = f"subsets/{dset_name}_subsets{n_subset}/{nimgs_class}imgs_class_{test_split}.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file = open(file_path, 'w')

    print(f"\nCreating {test_split} subset")
    for idx in tqdm(indices):
        img_label = test_data[idx][1].item()
        rel_img_path = test_data.dataset._input_array[test_data.indices[idx]]
        if img_label not in seen_classes: continue 
        file.write(rel_img_path + "\t" + str(img_label) + "\n")
    file.close()

exit(0)