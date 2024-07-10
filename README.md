# M-CD
This repository contains the code for "A Mamba-based Siamese Network for Remote Sensing Change Detection". Project page can be found [here](https://jayparanjape.github.io/M-CD/)

## Abstract
Change detection in remote sensing images is an essential tool for analyzing a region at different times. It finds varied applications in monitoring environmental changes, man-made changes as well as corresponding decision-making and prediction of future trends. Deep learning methods like Convolutional Neural Networks (CNNs) and Transformers have achieved remarkable success in detecting significant changes, given two images at different times. In this paper, we propose a Mamba-based Change Detector (M-CD) that segments out the regions of interest even better. Mamba-based architectures demonstrate linear-time training capabilities and an improved receptive field over transformers. Our experiments on four widely used change detection datasets demonstrate significant improvements over existing state-of-the-art (SOTA) methods.

## Environment File
Create a new conda environment with the config file given in the repository as follows:
```
conda env create --file=m-cd.yml
conda activate m-cd
```

Install Mamba as follows:
```
cd models/encoders/selective_scan && pip install . && cd ../../..
```

## General file descriptions
- configs/*.py - config files which control multiple parameters related to data training, logging etc.
- dataloader/changeDataset.py - dataset class defined here.
- models/* - model files available here
- train.py - driver file for training. Instructions below
- eval.py - driver file for evaluation. Instructions below

## Link to model checkpoints
Coming Soon

## Datasets

1. We test our models on four public Change Detection datasets:
    - [DSIFN-CD](https://www.dropbox.com/s/1lr4m70x8jdkdr0/DSIFN-CD-256.zip?dl=0)
    - [LEVIR-CD](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)
    - [WHU-CD](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)
    - [CDD](https://www.dropbox.com/s/ls9fq5u61k8wxwk/CDD.zip?dl=0)

    Please refer to the original dataset websites for more details. The links above containg the preprocessed splits, obtained from [DDPM-CD](https://github.com/wgcban/ddpm-cd)

2. If you are using your own datasets, please orgnize the dataset folder in the following structure:
    ```
    <root_folder>
    |-- A
        |-- <name1>.png
        |-- <name2>.png
        ...
    |-- B
        |-- <name1>.png
        |-- <name2>.png
        ...
    |-- gt
        |-- <name1>.png
        |-- <name2>.png
        ...
    |-- list
        |-- train.txt
        |-- val.txt
        |-- test.txt
    ```

    `train.txt/val.txt/test.txt` contains the names of items in training/validation/testing set, e.g.:

    ```
    <name1>
    <name2>
    ...
    ```

Please make sure to change the root folder in the config files available in the folder "configs". Also, if the files are in a format other than png, please specify the extension in the config.

For custom datasets, you would need to create a config file similar to the existing files in "config" folder.

## Training
1. Please download the pretrained [VMamba](https://github.com/MzeroMiko/VMamba) weights:

    - [VMamba_Tiny](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth).
    - [VMamba_Small](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmsmall_dp03_ckpt_epoch_238.pth).
    - [VMamba_Base](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmbase_dp06_ckpt_epoch_241.pth).

    <u> Please put them under `pretrained/vmamba/`. </u>


2. Config setting.

    Edit config file in the `configs` folder.    
    Change C.backbone to `sigma_tiny` / `sigma_small` / `sigma_base` to use the three versions of Sigma. 

3. Run multi-GPU distributed training:

    ```shell
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4  --master_port 29502 train.py -p 29502 -d 0,1,2,3 -n "dataset_name"
    ```

    Here, `dataset_name=dsifn/cdd/whu/levir`, referring to the four datasets.

4. You can also use single-GPU training:

    ```shell
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun -m --nproc_per_node=1 train.py -p 29501 -d 0 -n "dataset_name" 
    ```
5. Results will be saved in `log_final` folder.


## Evaluation
1. Run the evaluation by:

    ```shell
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"
    ```

    Here, `dataset_name=dsifn/cdd/whu/levir`, referring to the four datasets.\
    `epoch_number` refers to a number standing for the epoch number you want to evaluate with. You can also use a `.pth` checkpoint path directly for `epoch_number` to test for a specific weight.

2. If you want to use multi GPUs please specify multiple Device IDs:

    ```shell
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0,1,2,3,4,5,6,7" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"
    ```

3. Results will be saved in `log_final` folder.

## Acknowledgements
Our code is adapted from [Sigma](https://github.com/zifuwan/Sigma). We thanks the authors for their valuable contribution and well-written code. The data splits and preprocessing are taken from [DDPM-CD](https://github.com/wgcban/ddpm-cd). We thank the authors for motivating this topic and facilitating further research in this topic by providing easy access to the datasets.

## Citation
```
To be added
```
