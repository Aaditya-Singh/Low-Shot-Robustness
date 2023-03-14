# TODO: hardcoded configs in config yaml
Dataset=iwc
Model=msn_vits16
Type=pretrain
Tag=default

python main_distributed.py \
  --eval-type ${Type} \
  --fname configs/${Type}/${Model}_${Dataset}.yaml \
  --folder logs/submitit/${Model}/${Type}_${Dataset}/${Tag} \
  --partition long --gpu-type a40 \
  --nodes 1 --tasks-per-node 4 \
  --time 9360
