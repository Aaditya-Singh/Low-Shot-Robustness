# Code adapted from https://github.com/mlfoundations/wise-ft/

import os, sys
import argparse, pprint

import torch
import numpy as np
from tqdm import tqdm

import clip.clip as clip
import src.templates as templates
from src.classifier import LinearClassifier
from src.dataset_classnames import get_classnames

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save-type",
    type=str,
    default='zeroshot',
    help="zeroshot, wiseft",
)
parser.add_argument(
    "--save-path",
    type=str,
    default='pretrained/clip/imagenet_zeroshot_head.pth.tar'
)
parser.add_argument(
    "--zeroshot-weights",
    type=str,
    default=None,
    help="path to load zeroshot model weights",
)
parser.add_argument(
    "--finetuned-weights",
    type=str,
    default='pretrained/clip/clip_vitb16_inet_wiseft_full.pth.tar',
    help="path to load finetuned (with zeroshot head) model weights",
)
parser.add_argument(
    "--alpha",
    type=int,
    default=0.5,
    help="set alpha to 0/1 for zeroshot/finetuned checkpoint",
)
parser.add_argument(
    "--dataset",
    type=str,
    default='imagenet',
    help="imagenet, iwildcam, camelyon",
)
parser.add_argument(
    "--model-name",
    type=str,
    default='ViT-B/16',
)
parser.add_argument(
    "--device",
    type=str,
    default='cuda:0',
)
parser.add_argument(
    '--port', type=int,
    default=40111,
    help='port')

NUM_CLASSES, TEMPLATE = 0, None
pp = pprint.PrettyPrinter(indent=4)
torch.backends.cudnn.benchmark = True


class ImageEncoder(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()

        self.model, self.train_preprocess = clip.load(
            model, device, jit=False)

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)


def save_zeroshot_classifier(args):
    assert args.head_weights_path is not None
    global NUM_CLASSES, TEMPLATE
    if args.dataset == 'camelyon':
        NUM_CLASSES, TEMPLATE = 2, 'simple_template'
    elif args.dataset == 'iwildcam':
        NUM_CLASSES, TEMPLATE = 182, 'iwildcam_template'
    else:   # imagenet
        NUM_CLASSES, TEMPLATE = 1000, 'openai_imagenet_template'

    device = args.device
    torch.cuda.set_device(device)

    template = getattr(templates, TEMPLATE)
    image_encoder = ImageEncoder(args.model_name, device)
    clip_model = image_encoder.model
    logit_scale = clip_model.logit_scale

    clip_model.eval()
    clip_model.to(device)

    print(f'Saving zero-shot head weights for {args.dataset} dataset...')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(get_classnames(args.dataset)):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device) # tokenize

            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= logit_scale.exp()
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    emb_dim = 512 if args.model_name == 'ViT-B/16' else 1024
    linear_classifier = LinearClassifier(emb_dim, NUM_CLASSES, True, False)
    with torch.no_grad():
            linear_classifier.linear.weight.copy_(zeroshot_weights)

    save_dict = {
        'linear_classifier': linear_classifier.state_dict(),
    }
    torch.save(save_dict, args.save_path)
    return linear_classifier


def save_weightspace_ensemble(args):
    # NOTE: assumes encoder + head (zeroshot and finetuned) weights are saved offline
    assert args.zeroshot_weights is not None
    assert args.finetuned_weights is not None
    alpha = args.alpha
    zeroshot_state_dict = torch.load(args.zeroshot_weights, map_location='cpu')
    ft_state_dict = torch.load(args.finetuned_weights, map_location='cpu')
    wiseft_state_dict = {}
    for k, v in zeroshot_state_dict.items():
        if k not in ft_state_dict: continue
        print(f"Saving WiSE-FT weights for {k}...")
        wiseft_state_dict[k] = {}
        for n, zs_p in tqdm(zeroshot_state_dict[k].items()):
            if n in ft_state_dict[k]:
                ft_p = ft_state_dict[k][n]
            elif 'module.'+n in ft_state_dict[k]:
                ft_p = ft_state_dict[k]['module.'+n]
            else:
                raise Exception(f"Parameter {n} not found in key {k} of finetuned checkpoint.")
                exit(0)
            wiseft_state_dict[k][n] = (1 - alpha)*zs_p + alpha*ft_p
        print(f"Done!")
    torch.save(wiseft_state_dict, args.save_path)
    return wiseft_state_dict


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    os.environ["MASTER_PORT"] = str(args.port)
    if args.save_type == 'zeroshot':
        save_zeroshot_classifier(args)
    else:   # wiseft
        save_weightspace_ensemble(args)
    exit(0)