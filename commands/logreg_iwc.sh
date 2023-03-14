# TODO: hardcoded image folder and model name
SS=subsets1
FT=1percent
Eval=iwildcam
ValSplit=val
Model=deit_vitb16

python logistic_eval.py \
  --root-path-train /srv/share4/datasets/wilds/data/ \
  --image-folder-train iwildcam_v2.0/ \
  --subset-path subsets/${Eval}_${SS}/${FT}.txt \
  --root-path-test /srv/share4/datasets/wilds/data/ \
  --image-folder-test iwildcam_v2.0/ \
  --val-split ${ValSplit} \
  --model-name deit_base --fname deit_base_patch16.pth \
  --pretrained pretrained/deit/ \
  --penalty l2 --lambd 0.0025 \
  --log-file logs/${Model}/LR_${SS}/${FT}-${Eval}-${ValSplit}.log \
  --device cuda:0 --port 3101