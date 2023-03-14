# TODO: hardcoded configs in config yaml
Dataset=iwc
SS=subsets1
FT=1percent
Eval=id_val
Model=msn_vits16
Type=bslplpl

python main.py \
  --fname configs/eval/${Type}_${Dataset}.yaml \
  --log-file logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  --port 1234 --devices cuda:0 cuda:1