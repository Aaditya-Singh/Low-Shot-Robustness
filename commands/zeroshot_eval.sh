# TODO: hardcoded image folder
Eval=iwildcam
Model=ViT-B/16
Folder=clip_vitb16
ValSplit=id_val

python zeroshot_eval.py \
  --root-path-test /srv/share4/datasets/wilds/data/ \
  --image-folder-test iwildcam_v2.0/ --val-split ${ValSplit} \
  --pretrained pretrained/clip/${Eval}_zeroshot_head.pth.tar \
  --model-name ${Model} \
  --log-file logs/${Folder}/ZeroShot-${Eval}.log \
  --device cuda:0 --port 1234