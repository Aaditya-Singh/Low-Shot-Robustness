# TODO: hardcoded configs in config yaml
Dataset=cmlyn
SS=none
FT=cmlyn
Eval=id_val
Model=dino_vits16
Type=ft

python main.py \
  --fname configs/eval/${Type}_${Dataset}.yaml \
  --log-file logs/${Model}/lpft_${SS}/${FT}-${Eval}.log \
  --port 1234 --devices cuda:0 cuda:1