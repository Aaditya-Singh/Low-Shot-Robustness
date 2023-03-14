# TODO: hardcoded configs in config yaml
Dataset=cmlyn
SS=subsets1
FT=15000imgs_class
Eval=id_val
Model=msn_vitb16
Type=lineval

python main_distributed.py \
  --eval-type ${Type} \
  --fname configs/eval/${Type}_${Dataset}.yaml \
  --folder logs/submitit/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  --partition short --gpu-type a40 \
  --nodes 1 --tasks-per-node 2 \
  --time 2880
