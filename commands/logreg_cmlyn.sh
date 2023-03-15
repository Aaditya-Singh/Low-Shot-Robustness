SS=subsets1
FT=1500imgs_class
Eval=camelyon17
ValSplit=id_val
Model=msn_vits16

python logistic_eval.py \
  --root-path-train ../datasets/wilds/data/ \
  --image-folder-train camelyon17_v1.0/ \
  --subset-path subsets/${Eval}_${SS}/${FT}.txt \
  --root-path-test ../datasets/wilds/data/ \
  --image-folder-test camelyon17_v1.0/ \
  --val-split ${ValSplit} \
  --model-name deit_small --fname vits16_800ep.pth.tar \
  --pretrained pretrained/msn/ \
  --penalty l2 --lambd 0.0025 \
  --log-file logs/${Model}/LR_${SS}/${FT}-${Eval}-${ValSplit}.log \
  --device cuda:0 --port 1234