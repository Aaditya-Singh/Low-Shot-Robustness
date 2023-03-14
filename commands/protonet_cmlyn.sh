# TODO: hardcoded image folder and model name
SS=subsets1
FT=1500imgs_class
Eval=camelyon17
Model=msn_vits16

python protonet_eval.py \
  --root-path-train /srv/share4/datasets/wilds/data/ \
  --image-folder-train camelyon17_v1.0/ \
  --subset-path subsets/${Eval}_${SS}/${FT}.txt \
  --root-path-test /srv/share4/datasets/wilds/data/ \
  --image-folder-test camelyon17_v1.0/ \
  --val-split id_val \
  --model-name deit_small --fname vits16_800ep.pth.tar \
  --pretrained pretrained/msn/ \
  --log-file logs/${Model}/ProtoNet_${SS}/${FT}-${Eval}.log \
  --device cuda:0