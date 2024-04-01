<p align="center">
  <img src="/www/skellytour_banner.png">
</p>

# Skellytour: Automated Skeleton Segmentation from Whole-Body CT Images
## Key Features
- **Segments bones from any CT scan**
- **Based on the state-of-the-art [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet "nnUNetv2 on GitHub") package**
- **3 models with increasing numbers of labels (17, 38 or 60 labels), including a label for high-radiodensity objects (e.g. surgically implanted hardware)**
- **Subsegmentation model to segment bones into cortical and trabecular regions**
- **Trained using a high quality, manually segmented dataset of elderly multiple myeloma patients with low bone density and osteolytic lesions**
- **High accuracy; average Dice for medium model is 0.935**
- **GPU, CPU and MPS compute devices supported**
- **GPU with at least 8GB of RAM is recommended, but will work with CPU only**
- **Developed on Ubuntu using WSL, but should work on Windows, Linux and macOS**

## News
- **April 2024: updated models and code to nnUNetv2**
- **October 2023: initial release using nnUNetv1**

## Citing Skellytour
For more information and if you use Skellytour in your work, please cite our upcoming paper:
`Mann D.C., Rutherford M., Farmer P., Eichhorn J., Palot Manzil F.F., Wardell C.P. (2024). Skellytour: Automated Skeleton Segmentation from Whole-Body CT Images`

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Available Models](#available-models)
- [Description of Output](#description-of-output)
- [Getting Help](#getting-help)
- [Label List and Description](#label-list-and-description)

## Introduction
Segmenting bones from CT scans is required for a range of both clinical and research applications; e.g. diagnosing fractures, surgical planning, producing 3D models of bones, quantitative image analysis. However, segmentation is a time-consuming and laborious process. We present `Skellytour`, an easy-to-use tool for bone segmentation from CT scans. `Skellytour` is based on the state-of-the-art [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and was trained using a high quality, manually segmented dataset of elderly patients with low bone density and osteolytic lesions. Segmentations were verified by a board-certified radiologist and a board-certified nuclear medicine physician. Tested against an internal and two external datasets, the medium model achieved high metrics (Dice scores of 0.935, 0.936, 0.953 and Normalized Surface Distance of 0.993, 0.999, 0.990). It includes 3 models with increasing numbers of labels (17, 38 or 60 labels), with a separate label for high-radiodensity artifacts such as surgically implanted hardware. An optional model subsegments bones into cortical and trabecular regions.

## Installation
It is recommended but not required that you install Skellytour in a Conda environment using Python v3.9. You can find Conda installation instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html "Conda installation instructions").
```
## Create new Conda environment and activate it
conda create -y -n skellytour python=3.9
conda activate skellytour
```

Download and install Skellytour via GitHub using pip. All dependencies will automatically be installed:

```
git clone https://github.com/cpwardell/Skellytour.git
cd Skellytour/
python -m pip install .
```

You can test if Skellytour has installed correctly by running `skellytour` on the command line. It will print the help message and notify you if a GPU is detected and how many are available.

```
$ skellytour
error: the following arguments are required: -i
usage: skellytour [-h] -i I [-o O] [-m {low,medium,high}] [-c C] [-d {gpu,cpu,mps}] [-g G] [--overwrite] [--nopp] [--subseg] [--fast]

Skellytour: Bone Segmentation from CT scans

optional arguments:
  -h, --help            show this help message and exit
  -i I                  path to input NIfTI file (default: None)
  -o O                  path to output directory (default: .)
  -m {low,medium,high}  model to use; can be low (17 labels), medium (38 labels, default), high (60 labels) (default: medium)
  -c C                  number of CPU cores to use for preprocessing and postprocessing (default: 6)
  -d {gpu,cpu,mps}      compute device to use (default: gpu)
  -g G                  GPU to use (default: 0)
  --overwrite           overwrite previous results if they exist (default: False)
  --nopp                skip postprocessing on predicted segmentations (default: False)
  --subseg              perform subsegmentation, to predict trabecular and cortical labels (default: False)
  --fast                perform segmentation tasks with a single fold, not the full ensemble model. Not recommended (default: False)

1 GPU detected
```

## Usage
Skellytour requires only a gzipped NIfTI file of a CT scan as input. Note that you should provide a path to the actual file, *not* the directory containing it. The default usage is:
```
skellytour -i /path/to/input/nifti.nii.gz
```
This will run the `medium` model with 38 labels and write output to the current working directory. When a model is run for the first time, Skellytour will download the pretrained model from GitHub and store it in a hidden directory in the current user's home directory e.g. `/home/user/.skellytour`. Each model requires approximately 1.2 GB of storage space.

Below is a more complex command that would produce a `high` (60 label) bone segmentation and additional subsegmentation with trabecular and cortical labels, using the second GPU:
```
skellytour -i /path/to/input/nifti.nii.gz -m high -g 1 --subseg
```

The following arguments are available:

Argument | Description
--- | ---
**`-h, --help`** | show this help message and exit
**`-i`** | path to input NIfTI file (default: None)
**`-o`** | path to output directory (default: .)
**`-m`** | model to use; can be low (17 labels), medium (38 labels, default), high (60 labels) (default: medium)
**`-c`** | number of CPU cores to use for preprocessing and postprocessing (default: 6)
**`-d`** | compute device to use; either gpu, cpu or mps (default: gpu)
**`-g`** | GPU to use if you have multiple; 0 is the first, 1 the second, etc (default: 0)
**`--overwrite`** | overwrite previous results if they exist (default: False)
**`--nopp`** | skip postprocessing on predicted segmentations (default: False)
**`--subseg`** | perform subsegmentation, assigning trabecular and cortical labels (default: False)
**`--fast`** | perform segmentation tasks with a single fold, not the full ensemble model. Not recommended (default: False)

## Available Models
There are 3 main models and a subsegmentation model. The main models (`low`,`medium`, `high`) have increasing numbers of labels and are detailed in the `Label List and Description` section of this document. The subsegmentation model runs after the main model if invoked with the `--subseg` flag and will segment the bones into trabecular and cortical regions.
<p align="center">
  <img src="/www/segmentation_schemes.png">
</p>

## Description of Output
While running, Skellytour will print messages to the screen updating users on progress. Output files are gzipped NIfTI files containing multilabel segmentations. As an example, the command below will run the medium and subsegmentation models and write output to the `outputdir` directory:
```
skellytour -i example.nii.gz -o outputdir --subseg
```

Output file | Description
--- | ---
**log.txt** | Text file containing all details of the process
**example_medium.nii.gz** | Raw segmentation produced by the medium model
**example_medium_postprocessed.nii.gz** | Postprocessed medium model segmentation
**example_medium_postprocessed_subseg.nii.gz** | Raw segmentation produced by the subsegmentation model
**example_medium_postprocessed_subseg_postprocessed.nii.gz** | Postprocessed subsegmentation model segmentation


## Getting Help
If you find an issue not covered in this document, or want to request a new feature or model, please open a new issue on GitHub.

## Label List and Description
A complete list of all numeric labels in each model is below.

Labelling Scheme | Numeric Label | Description
--- | --- | ---
**High** | 1 | SKULL
**High** | 2 | PELVIS
**High** | 3 | STERNUM
**High** | 4 | LEFT_FEMUR
**High** | 5 | RIGHT_FEMUR
**High** | 6 | LEFT_HUMERUS
**High** | 7 | RIGHT_HUMERUS
**High** | 8 | LEFT_SCAPULA
**High** | 9 | RIGHT_SCAPULA
**High** | 10 | LEFT_CLAVICLE
**High** | 11 | RIGHT_CLAVICLE
**High** | 12 | LEFT_RIB_1
**High** | 13 | LEFT_RIB_2
**High** | 14 | LEFT_RIB_3
**High** | 15 | LEFT_RIB_4
**High** | 16 | LEFT_RIB_5
**High** | 17 | LEFT_RIB_6
**High** | 18 | LEFT_RIB_7
**High** | 19 | LEFT_RIB_8
**High** | 20 | LEFT_RIB_9
**High** | 21 | LEFT_RIB_10
**High** | 22 | LEFT_RIB_11
**High** | 23 | LEFT_RIB_12
**High** | 24 | RIGHT_RIB_1
**High** | 25 | RIGHT_RIB_2
**High** | 26 | RIGHT_RIB_3
**High** | 27 | RIGHT_RIB_4
**High** | 28 | RIGHT_RIB_5
**High** | 29 | RIGHT_RIB_6
**High** | 30 | RIGHT_RIB_7
**High** | 31 | RIGHT_RIB_8
**High** | 32 | RIGHT_RIB_9
**High** | 33 | RIGHT_RIB_10
**High** | 34 | RIGHT_RIB_11
**High** | 35 | RIGHT_RIB_12
**High** | 36 | C1
**High** | 37 | C2
**High** | 38 | C3
**High** | 39 | C4
**High** | 40 | C5
**High** | 41 | C6
**High** | 42 | C7
**High** | 43 | T1
**High** | 44 | T2
**High** | 45 | T3
**High** | 46 | T4
**High** | 47 | T5
**High** | 48 | T6
**High** | 49 | T7
**High** | 50 | T8
**High** | 51 | T9
**High** | 52 | T10
**High** | 53 | T11
**High** | 54 | T12
**High** | 55 | L1
**High** | 56 | L2
**High** | 57 | L3
**High** | 58 | L4
**High** | 59 | L5
**High** | 60 | ARTIFACTS
**Medium** | 1 | SKULL
**Medium** | 2 | PELVIS
**Medium** | 3 | STERNUM
**Medium** | 4 | LEFT_FEMUR
**Medium** | 5 | RIGHT_FEMUR
**Medium** | 6 | LEFT_HUMERUS
**Medium** | 7 | RIGHT_HUMERUS
**Medium** | 8 | LEFT_SCAPULA
**Medium** | 9 | RIGHT_SCAPULA
**Medium** | 10 | LEFT_CLAVICLE
**Medium** | 11 | RIGHT_CLAVICLE
**Medium** | 12 | LEFT_RIBS
**Medium** | 13 | RIGHT_RIBS
**Medium** | 14 | C1
**Medium** | 15 | C2
**Medium** | 16 | C3
**Medium** | 17 | C4
**Medium** | 18 | C5
**Medium** | 19 | C6
**Medium** | 20 | C7
**Medium** | 21 | T1
**Medium** | 22 | T2
**Medium** | 23 | T3
**Medium** | 24 | T4
**Medium** | 25 | T5
**Medium** | 26 | T6
**Medium** | 27 | T7
**Medium** | 28 | T8
**Medium** | 29 | T9
**Medium** | 30 | T10
**Medium** | 31 | T11
**Medium** | 32 | T12
**Medium** | 33 | L1
**Medium** | 34 | L2
**Medium** | 35 | L3
**Medium** | 36 | L4
**Medium** | 37 | L5
**Medium** | 38 | ARTIFACTS
**Low** | 1 | SKULL
**Low** | 2 | PELVIS
**Low** | 3 | STERNUM
**Low** | 4 | LEFT_FEMUR
**Low** | 5 | RIGHT_FEMUR
**Low** | 6 | LEFT_HUMERUS
**Low** | 7 | RIGHT_HUMERUS
**Low** | 8 | LEFT_SCAPULA
**Low** | 9 | RIGHT_SCAPULA
**Low** | 10 | LEFT_CLAVICLE
**Low** | 11 | RIGHT_CLAVICLE
**Low** | 12 | LEFT_RIBS
**Low** | 13 | RIGHT_RIBS
**Low** | 14 | CERVICAL_VERTEBRAE
**Low** | 15 | THORACIC_VERTEBRAE
**Low** | 16 | LUMBAR_VERTEBRAE
**Low** | 17 | ARTIFACTS
**Subsegmentation** | 1 | TRABECULAR BONE
**Subsegmentation** | 2 | CORTICAL BONE




