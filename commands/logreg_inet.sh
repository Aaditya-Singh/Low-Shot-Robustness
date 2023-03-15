SS=subsets1
FT=1percent
Eval=INv2
Model=swav_rn50w2

python logistic_eval.py \
  --root-path-train ../datasets/ImageNet/ \
  --image-folder-train imagenet/ \
  --subset-path subsets/imagenet_${SS}/${FT}.txt \
  --root-path-test ../datasets/ImageNet/ \
  --image-folder-test imagenetv2/ \
  --model-name resnet50w2 --fname swav_RN50w2_400ep_pretrain.pth.tar \
  --pretrained pretrained/swav/ \
  --log-file logs/${Model}/LR_${SS}/${FT}-${Eval}.log \
  --device cuda:0 --port 1234