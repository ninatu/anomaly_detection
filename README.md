# Anomaly Detection in Medical Imaging With Deep Perceptual Autoencoders — Pytorch Implementation

[![License][license-shield]][license-url]

**Anomaly Detection in Medical Imaging With Deep Perceptual Autoencoders**<br>
Nina Tuluptceva, Bart Bakker, Irina Fedulova, Heinrich Schulz, and Dmitry V. Dylov.<br>
2021<br>

[https://ieeexplore.ieee.org/abstract/document/9521238](https://ieeexplore.ieee.org/abstract/document/9521238)

[https://arxiv.org/abs/2006.13265](https://arxiv.org/abs/2006.13265)

[comment]: <> (*Anomaly detection is the problem of recognizing abnormal inputs based on the seen examples of normal data. Despite recent advances of deep learning in recognizing image anomalies, these methods still prove incapable of handling complex images, such as those encountered in the medical domain. Barely visible abnormalities in chest X-rays or metastases in lymph nodes on the scans of the pathology slides resemble normal images and are very difficult to detect. To address this problem, we introduce a new powerful method of image anomaly detection. It relies on the classical autoencoder approach with a re-designed training pipeline to handle high-resolution, complex images, and a robust way of computing an image abnormality score. We revisit the very problem statement of fully unsupervised anomaly detection, where no abnormal examples are provided during the model setup. We propose to relax this unrealistic assumption by using a very small number of anomalies of confined variability merely to initiate the search of hyperparameters of the model. We evaluate our solution on natural image datasets with a known benchmark, as well as on two medical datasets containing radiology and digital pathology images. The proposed approach suggests a new strong baseline for image anomaly detection and outperforms state-of-the-art approaches in complex pattern analysis tasks.*)

This is the official implementation of "Anomaly Detection in Medical Imaging With Deep Perceptual Autoencoders. 
It includes experiments reported in the paper.

## Structure of Project 
    anomaly_detection - python package; implementations of 
                                    deep_geo: Deep Anomaly Detection Using Geometric Transformations  (https://arxiv.org/abs/1805.10917)
                                    deep_if: Towards Practical Unsupervised Anomaly Detection on Retinal Images (https://link.springer.com/chapter/10.1007/978-3-030-33391-1_26)
                                    piad: Perceptual Image Anomaly Detection (https://arxiv.org/abs/1909.05904)
                                    dpa: Anomaly Detection with Deep Perceptual Autoencoders (https://arxiv.org/abs/2006.13265).    
    configs - yaml configs to reproduce experiments, reported in the paper
        └───deep_geo - configs used to train and eval deep_geo models
        |   |   train_example.yaml -- Example of train config of deep_geo model with a description of all params
        |   |   eval_example.yaml -- -- Example of a config for evaluation of deep_geo model with a description of all params
        │   └───camelyon16
        |   |   └───meta - "meta" configs with missed values (to generate configs for hyperparameter search)
        |   |   │   |   ...
        |   |   └───final - configs used in final experiments
        |   |   │   └───reproduce -- configs with default hyperparameters provided by the authors of the method (to reproduce the papers' results)
        |   |   │   │    ... 
        |   |   │   └───with_cv -- configs with hyperparameters found by cross-validation search (see paper for more detail)
        |   |   │   │    ...
        │   └───cifar10
        |   |    ...
        │   └───nih 
        |   |    ...
        │   └───svhn
        |   |    ...
        └───deep_if - configs used to train and eval deep_if models
        |   ...
        └───dpa - configs used to train dpa models
        |   ...
        └───piad - configs used to train piad models
        |   ...
     camelyon16_preprocessing -- Scripts for preprocessing Camelyon16 dataset
     folds -- Folds used in cross-validation, train/test split of NIH and Camelyon16, validation info 
     
## Installation 

Requirements: `Python3.6`
 
You can install miniconda environment(version 4.5.4):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
bash Miniconda3-4.5.4-Linux-x86_64.sh
export PATH="{miniconda_root/bin}:$PATH
```

Installation:
```bash
pip install -r requirements.txt
pip install -e .
```

## Training and Evaluation 

The paper includes experiments on CIFAR10, SVHN, Camelyon16, and NIH datasets. 

To get started with CIFAR10 and SVHN, data downloading is NOT required
(we used torchvision.datasets implementation of these datasets).

To work with Camelyon16, and NIH datasets, see section `Data Preprocessing`.

For each model: 
* deep_geo: Deep Anomaly Detection Using Geometric Transformations  (https://arxiv.org/abs/1805.10917)
* deep_if: Towards Practical Unsupervised Anomaly Detection on Retinal Images (https://link.springer.com/chapter/10.1007/978-3-030-33391-1_26)
* piad: Perceptual Image Anomaly Detection (https://arxiv.org/abs/1909.05904)
* dpa: Anomaly Detection with Deep Perceptual Autoencoders (https://arxiv.org/abs/2006.13265).    

there are `main.py` scripts in corresponding directory in anomaly detection/{deep_geo,deep_if/piad,dpa}

and examples of train/evaluate configs in corresponding files in configs/{deep_geo,deep_if/piad,dpa}/{train_example/eval_example}.yaml
See the configs for more details. 


Try:
```bash
python anomaly_detection/dpa/main.py train_eval configs/dpa/train_wo_pg_example.yaml configs/dpa/eval_wo_pg_example.yaml
```

To reproduce all experiments of the paper, run:

```bash
python run_experiments.py
```

Or specify a subset:
```bash
python run_experiments.py --model dpa deep_if --datasets camelyon16 --ablation
```


##  Hyperparameter Tuning (cross-validation) 

Cross-validation folds used in the paper are stored in `./folds/folds/`.
Information about classes and images used for validation is in `./folds/validation_classes/`.
Train/test split for Camelyon16 and NIH (AP, PA, a subset) dataset is in `./folds/train_test_split/`.

To generate cross-validation folds by yourself, use scripts from the folder ` anomaly_detection/utils/preprocessing/create_folds/`.

For example:
```bash
python anomaly_detection/utils/preprocessing/create_folds/cifar10.py -o ./my_folds/folds -n 3
```

## Data Preprocessing 

### Camelyon16

Camelyon16 is a challenge conducted in 2016 of automated detection of metastases 
in hematoxylin and eosin (H&E) stained whole-slide images of lymph node sections.

See [Offical Challenge Website](https://camelyon16.grand-challenge.org) for more details. 

Preprocessing steps:

1. Download data of camelyon16 challenge [link](https://camelyon16.grand-challenge.org/Data/), 
 store it, for example,  in `./data/data/camelyon16_original` directory
    ```
    ./data/data/camelyon16_original
    │   ...
    └───training
    │   │   lesion_annotations.zip (111 xml)
    │   │───normal (159 tif)
    │       │   normal_001.tif
    │       │   ...
    │   │───tumor (111 tif)
    │       │   tumor_001.tif
    │       │   ...
    └───testing
    │   │   lesion_annotations.zip (48 xml)
    │   │   reference.csv (129 csv)
    │   │───images (129 tif)
    │       │   test_001.tif
    │       │   ...
    │   ...
    ```
2. Unzip both `lesion_annotations.zip` files
3. Build and run docker using, see `camelyon16_preprocessing` (put correct paths to `camelyon16_preprocessing/docker/run.sh`).
Or install openslide into your system.
    ```bash 
    cd camelyon16_preprocessing/docker
    bash build.sh
    bash run.sh
    ```
4. Perform preprocessing:
    ```bash
    /opt/anaconda/bin/python /scripts/1_convert_annotation_to_json.py
    /opt/anaconda/bin/python /scripts/2_create_tumor_masks.py
    /opt/anaconda/bin/python /scripts/3_generate_normal_patches_x40.py
    /opt/anaconda/bin/python /scripts/4_generate_tumor_patches_x40.py
    /opt/anaconda/bin/python /scripts/5_select_stain_normalization_image.py
    /opt/anaconda/bin/python /scripts/6_normalize_stain.py
    /opt/anaconda/bin/python /scripts/7_create_train_test_split.py
    /opt/anaconda/bin/python /scripts/8_merge_images.py
    /opt/anaconda/bin/python /scripts/9_resize_to_x20_and_x10.py
    ```

    1. Convert the xml-annotation files into json-format
    2. Create masks for tumor images (from json annotations)
    3. Generate normal patches (with the level of magnification x40) from the train split and the test split
    4. Generate tumor patches (with the level of magnification x40) from the train split and the test split
    5. Save crop from a source image as the "target" of stain normalization
    6. Perform stain normalization of all patches using script `normalize_stain.py`
    7. Create a train/test split (just create lists of the generated patches)
    8. Move all patches into one folder
    9. Create resized copies of patches with level of magnification x20 and x10



### NIH 

1. Download NIH data [link](https://nihcc.app.box.com/v/ChestXray-NIHCC)
    ```
    NIH
    │   ... 
    |   Data_Entry_2017.csv
    │	train_val_list.txt
    |   test_list.txt
    └───images
    │   │   batch_downlaad_zips.py 
    │   │   images_001.tar.gz
    |   |   images_002.tar.gz
    |   |   ...
    ```
2. Unzip all images (save it, for example, in `./data/data/nih/` folder)
3. Resize images to the resolution 300x300 (for faster loading) 
    ```bash 
    python anomaly_detection/utils/preprocessing/nih_resize.py
    ```
4. Create a train/test split (just filter train/test lists for each view: AP, PA)
    ```bash 
    python anomaly_detection/utils/preprocessing/create_folds/cifar10.py [-h] -i CIFAR10_ROOT -o OUTPUT_ROOT [-n N_FOLDS]
    ```

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://github.com/ninatu/mood_challenge/blob/master/LICENSE

## Cite

If you use this code in your research, please cite
```bibtex
@ARTICLE{9521238,
  author={Shvetsova, Nina and Bakker, Bart and Fedulova, Irina and Schulz, Heinrich and Dylov, Dmitry V.},
  journal={IEEE Access}, 
  title={Anomaly Detection in Medical Imaging With Deep Perceptual Autoencoders}, 
  year={2021},
  volume={9},
  number={},
  pages={118571-118583},
  doi={10.1109/ACCESS.2021.3107163}}
```