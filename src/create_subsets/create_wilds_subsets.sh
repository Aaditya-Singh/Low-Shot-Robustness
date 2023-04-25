Dataset=camelyon17
Subset=7500
# Dataset=iwildcam
# Subset=10

if [ ${Dataset} == 'camelyon17' ]; then
  python src/create_subsets/wilds_balanced_subsets.py \
    --file-dir subsets/${Dataset}_subsets1/ \
    --data-path ../datasets/wilds/data \
    --dataset ${Dataset} \
    --subset ${Subset}

else
  python src/create_subsets/wilds_fraction_subsets.py \
    --file-dir subsets/${Dataset}_subsets1/ \
    --data-path ../datasets/wilds/data \
    --dataset ${Dataset} \
    --subset ${Subset}
fi