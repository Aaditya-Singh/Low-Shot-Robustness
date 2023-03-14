from tqdm import tqdm
from wilds import *
import os, json, torch, numpy as np
import torchvision.transforms as transforms

IWC_CLASS_COUNTS_PATH = 'src/iwildcam_class_counts.json'
TEST_SPLITS = []
iwc_class_counts = json.load(open(IWC_CLASS_COUNTS_PATH))

# Set random seed for reproducibility
n_subset, n_percent, lower_limit = 1, 5, 1
np.random.seed(n_subset)

# Load the full dataset, and download it if necessary
dset_name = "iwildcam"
data_path = "/srv/datasets/wilds/data"
full_dset = get_dataset(dataset=dset_name, root_dir=data_path, download=False)

################ Create training subset ################
train_data = full_dset.get_subset(
    split = "train", frac = 1, transform = transforms.ToTensor()
)

# Dump the relative image paths and labels in a text file
seen_classes = dict()
indices = [i for i in range(len(train_data))] # np.random.shuffle(indices)
file_path = f"subsets/{dset_name}_subsets{n_subset}/{n_percent}percent_train.txt"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
file = open(file_path, 'w')

print("\nCreating train subset")
for idx in tqdm(indices):
    img_label = train_data[idx][1].item()
    # > mfw when fixing the relative path bug ✍️🗿
    rel_img_path = train_data.dataset._input_array[train_data.indices[idx]]
    true_class_count = iwc_class_counts[str(img_label)]
    required_count = max(1, true_class_count * n_percent // 100)
    if true_class_count < lower_limit: continue
    elif img_label not in seen_classes:
        seen_classes[img_label] = 1
    elif seen_classes[img_label] < required_count:
        seen_classes[img_label] += 1
    else: continue
    file.write(rel_img_path + "\t" + str(img_label) + "\n")

print(f"\nImages obtained from classes with at least {lower_limit} number of images")
print(f"Count of such images with {n_percent}% fraction compared to original: \n", \
    dict(sorted(seen_classes.items())))
file.close()

################ Create test subsets ################
for test_split in TEST_SPLITS:    
    test_data = full_dset.get_subset(
        split = test_split, transform = transforms.ToTensor()
    )

    # Dump the relative image paths and labels in a text file
    indices = [i for i in range(len(test_data))] # np.random.shuffle(indices)
    file_path = f"subsets/{dset_name}_subsets{n_subset}/{n_percent}percent_{test_split}.txt"
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