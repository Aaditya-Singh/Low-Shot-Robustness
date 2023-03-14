# TODO: hardcoded configs in config yaml
Dataset=inet
SS=none
FT=full
Eval=IN1k
Model=msn_vitb16
Type=lineval

python main_distributed.py \
  --eval-type ${Type} \
  --fname configs/eval/${Type}_${Dataset}.yaml \
  --folder logs/submitit/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  --partition long --gpu-type a40 \
  --nodes 1 --tasks-per-node 4 \
  --time 7200
