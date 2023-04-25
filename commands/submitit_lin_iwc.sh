Dataset=iwc
SS=none
FT=full
Eval=id_val
Model=msn_vits16
Type=ft

python main_distributed.py \
  --eval-type ${Type} \
  --fname configs/${Type}_${Dataset}.yaml \
  --folder logs/submitit/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  --partition short --gpu-type 2080_ti \
  --nodes 1 --tasks-per-node 4 \
  --time 2880
