Dataset=iwc
SS=none
FT=full
Eval=id_val
Model=sup_rn50
Type=ft

python main.py \
  --fname configs/${Type}_${Dataset}.yaml \
  --log-file logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  --port 1234 --devices cuda:0