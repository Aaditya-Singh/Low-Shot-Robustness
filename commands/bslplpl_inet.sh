Dataset=inet
SS=subsets1
FT=1imgs_class
Eval=IN1k
Model=msn_vitb16
Type=bslplpl

python main.py \
  --fname configs/${Type}_${Dataset}.yaml \
  --log-file logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  --port 1234 --devices cuda:0 cuda:1