Dataset=camelyon
Nickname=cmlyn
Subset=7500imgs_class
Model=ViT-B/16
Type=wiseft
Alpha=0.5

if [ ${Type} == 'zeroshot' ]; then
  python save_wiseft_weights.py \
    --save-type ${Type} \
    --save-path pretrained/clip/${Dataset}_zeroshot_head.pth.tar \
    --dataset ${Dataset} \
    --model-name ${Model} \
    --device cuda:0 --port 1234

else
  python save_wiseft_weights.py \
    --save-type ${Type} --alpha ${Alpha} \
    --save-path pretrained/clip/${Dataset}_${Type}_${Subset}_${Alpha}.pth.tar \
    --zeroshot-weights pretrained/clip/ViTB16_zeroshot${Nickname}.pth.tar \
    --finetuned-weights pretrained/clip/clip_vitb16_${Nickname}_wiseft_${Subset}.pth.tar \
    --dataset ${Dataset} \
    --model-name ${Model} \
    --device cuda:0 --port 1234
fi