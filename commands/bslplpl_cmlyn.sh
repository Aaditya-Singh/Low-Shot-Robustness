Dataset=cmlyn
SS=subsets1
FT=1500imgs_class
Eval=id_val
Model=deit_vitb16_in21k
Type=bslplpl

python main.py \
  --fname configs/${Type}_${Dataset}.yaml \
  --log-file logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  --port 1234 --devices cuda:0 cuda:1