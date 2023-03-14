# TODO: hardcoded image folder and model name
SS=subsets1
FT=1imgs_class
Eval=IN1k
Model=msn_vitb16

python protonet_eval.py \
  --root-path-train /srv/share4/datasets/ImageNet/ \
  --image-folder-train imagenet/ \
  --subset-path subsets/imagenet_${SS}/${FT}.txt \
  --root-path-test /srv/share4/datasets/ImageNet/ \
  --image-folder-test imagenet/ \
  --model-name deit_base --fname vitb16_600ep.pth.tar \
  --pretrained pretrained/msn/ \
  --log-file logs/${Model}/ProtoNet_${SS}/${FT}-${Eval}.log \
  --device cuda:0