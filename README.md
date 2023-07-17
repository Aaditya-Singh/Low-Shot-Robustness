# Benchmarking Low-Shot Robustness To Natural Distribution Shifts

This repository contains the code for our ICCV 2023 paper: [Benchmarking Low-Shot Robustness To Natural Distribution Shifts](https://arxiv.org/abs/2304.11263). To clone the full repository along with the submodules, the following command can be used.
```
git clone --recurse-submodules https://github.com/Aaditya-Singh/Low-Shot-Robustness.git
```

![Results](LSR.png)


## Requirements
* Python 3.8 or newer (preferably through [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html)) and [Cyanure](http://thoth.inrialpes.fr/people/mairal/cyanure/welcome.html#installation).
* Install [WILDS benchmark](https://github.com/p-lambda/wilds) as a package and [other requirements](https://github.com/p-lambda/wilds#requirements).


## Datasets and low-shot subsets
* Please refer to [WiSE-FT](https://github.com/mlfoundations/wise-ft/blob/master/datasets.md) and [WILDS benchmark](https://github.com/p-lambda/wilds#data) for downloading the datasets.
* The low-shot subsets used in the paper can be found in the [subsets](https://github.com/Aaditya-Singh/Low-Shot-Robustness/tree/main/subsets) directory.
* Code for creating such low-shot subsets can be found in the [create_subsets](https://github.com/Aaditya-Singh/Low-Shot-Robustness/tree/main/src/create_subsets) directory.


## Standard models (pre-trained on ImageNet)
<table>
  <tr>
    <td> MSN checkpoints </td>
    <td><a href="https://github.com/facebookresearch/msn#pre-trained-models">download</a></td>
  </tr>
  <tr>
    <td> DINO checkpoints </td>
    <td><a href="https://github.com/facebookresearch/dino#pretrained-models">download</a></td>
  </tr>
  <tr>
    <td> DEIT checkpoints </td>
    <td><a href="https://github.com/facebookresearch/deit/blob/main/README_deit.md">download</a></td>
  </tr>
  <tr>
    <td> SwAV checkpoints </td>
    <td><a href="https://github.com/facebookresearch/swav#model-zoo">download</a></td>
  </tr>
</table>


## [CLIP](https://github.com/openai/CLIP) ViTB-16 with zero-shot head weights
<table>
  <tr>
    <td> ImageNet </td>
    <td><a href="https://www.dropbox.com/s/93u1tfow7ezmivg/ViTB16_zeroshotinet.pth.tar?dl=0">download</a></td>
  </tr>
  <tr>
    <td> iWildCam </td>
    <td><a href="https://www.dropbox.com/s/m52vsdw7e26xfzj/ViTB16_zeroshotiwc.pth.tar?dl=0">download</a></td>
  </tr>
  <tr>
    <td> Camelyon </td>
    <td><a href="https://www.dropbox.com/s/34xxd2x8vvlhdzp/ViTB16_zeroshotcmlyn.pth.tar?dl=0">download</a></td>
  </tr>
</table>


## Training and Testing

The bash commands used for fine-tuning can be found in the [commands](https://github.com/Aaditya-Singh/Low-Shot-Robustness/tree/main/commands) directory. Methods other than Logistic Regression and Mean Centroid Classifier additionally make use of the [config yamls](https://github.com/Aaditya-Singh/Low-Shot-Robustness/tree/main/configs). We summarize some of the important flags and keys for experimentation below.

* `root_path_train/test`: Specify the root directory containing the images for training or testing, e.g. `../datasets/`
* `image_folder_train/test`: Specify the directory containing the images for the different classes, e.g. `imagenet/`
* `val_split`: Set to `id_val` for training and `val` for out-of-domain (OOD) testing for WILDS datasets.
* `training`: Set to `true` for training and `false` for evaluation.
* `finetuning`: Set to `true` for full fine-tuning and `false` for training only the classifier.
* `eval_type`: Should be set to `bslplpl` for Baseline++. Default is `lineval`.
* `folder` and `pretrained_path`: Specify the directory and path to save and load model weights.

For more details and parameters than the ones provided here, please refer to the `--help` option. Details for full fine-tuning on ImageNet can be found in our [MAE](https://github.com/Aaditya-Singh/MAE) codebase.


## Robustness Interventions

This codebase supports [LP-FT](https://arxiv.org/abs/2202.10054) and [WiSE-FT](https://github.com/mlfoundations/wise-ft) interventions. Note that the same [general instructions](https://github.com/Aaditya-Singh/Low-Shot-Robustness/#training-and-testing) are also applicable for these interventions. We summarize some other important details below.

- For CLIP, the `clip` model should be [loaded](https://github.com/openai/CLIP#cliploadname-device-jitfalse) and `clip.visual` weights should be saved offline.
- CLIP's zero-shot head weights can be saved with the command provided [here](https://github.com/Aaditya-Singh/Low-Shot-Robustness/blob/main/commands/save_wiseft_weights.sh).
- Alternatively, the full set of weights (encoder and zero-shot head) for ViTB-16 can be found [here](https://github.com/Aaditya-Singh/Low-Shot-Robustness#clip-vitb-16-with-zero-shot-head-weights).
- Set `finetuning` to `true` and `eval_type` to `zeroshot` for full fine-tuning with these weights.
- This [command](https://github.com/Aaditya-Singh/Low-Shot-Robustness/blob/main/commands/save_wiseft_weights.sh) with `Type=wiseft` can be used to save WiSE-FT weights after full fine-tuning.

Please refer to our [RobustViT](https://github.com/Aaditya-Singh/RobustViT) and [Model Soups](https://github.com/ksarangmath/model-soups) codebases for additional interventions. We also provide them as submodules in this repository. The command for cloning the full repository is provided [here](https://github.com/Aaditya-Singh/Low-Shot-Robustness/#benchmarking-low-shot-robustness-to-natural-distribution-shifts).


## Bibtex

Please consider citing our paper if you find this repository helpful:
```
@misc{singh2023benchmarking,
      title={Benchmarking Low-Shot Robustness to Natural Distribution Shifts}, 
      author={Aaditya Singh and Kartik Sarangmath and Prithvijit Chattopadhyay and Judy Hoffman},
      year={2023},
      eprint={2304.11263},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## References

We follow these repositories and thank the authors for open-sourcing their code.

- [1]: [Masked Siamese Networks](https://github.com/facebookresearch/msn)
- [2]: [Masked Autoencoders](https://github.com/facebookresearch/mae)
