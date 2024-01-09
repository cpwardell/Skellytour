<p align="center">
  <img src="/www/skellytour_banner.png">
</p>

# Skellytour: Bone Segmentation from Whole-Body CT scans
## Key Features
- **Segments bones from any CT scan**
- **Based on the state-of-the-art nnU-Net package**
- **3 models with increasing numbers of labels (17, 38 or 60 labels), including a label for high-radiodensity artifacts**
- **Subsegmentation model to segment bones into cortical and trabecular regions**
- **Trained using a high quality, manually segmented dataset of elderly patients with low bone density and osteolytic lesions**
- **High accuracy; average Dice for medium model is 0.935**
- **GPU with 8GB of RAM is recommended, but will work with CPU only**

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
Segmenting bones from CT scans is required for a range of both clinical and research applications; e.g. diagnosing fractures, surgical planning, producing 3D models of bones, quantitative image analysis. However, segmentation is a time-consuming and laborious process. We present `Skellytour`, an easy-to-use tool for bone segmentation from CT scans. `Skellytour` is based on the state-of-the-art [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and was trained using a high quality, manually segmented dataset of elderly patients with low bone density and osteolytic lesions. Segmentations were verified by a board-certified radiologist and a board-certified nuclear medicine physician. Tested against an internal and two external datasets, the medium model achieved high metrics (Dice scores of 0.935, 0.936, 0.953 and Normalized Surface Distance of 0.993, 0.999, 0.990). It includes 3 models with increasing numbers of labels (17, 38 or 60 labels), and includes a separate label for high-radiodensity artifacts such as surgically implanted hardware. It includes an optional model to subsegment bones into cortical and trabecular regions.

## Installation
It is recommended but not required that you install Skellytour in a conda environment using Python v3.9:
```
## Create new conda environment and activate it
conda create -y -n skellytour python=3.9
conda activate skellytour
```

Download and install Skellytour via GitHub using pip.  All dependencies will automatically be installed:

```
git clone https://github.com/cpwardell/Skellytour.git
cd Skellytour/
python -m pip install .
```

You can test if Skellytour has installed correctly by running `skellytour` on the command line.  It will print the help message and notify you if a GPU is detected.

```
$ skellytour
error: the following arguments are required: -i
usage: skellytour [-h] -i I [-o O] [-m M] [--overwrite] [--nopp] [--subseg] [--fast]

Skellytour: Bone Segmentation from CT scans

options:
  -h, --help   show this help message and exit
  -i I         path to input NIfTI file (default: None)
  -o O         path to output directory (default: .)
  -m M         model to use; can be low (17 labels), medium (38 labels, default), high (60 labels) (default: medium)
  --overwrite  overwrite previous results if they exist (default: False)
  --nopp       skip postprocessing on predicted segmentations (default: False)
  --subseg     perform subsegmentation, assigning trabecular and cortical labels (default: False)
  --fast       perform segmentation tasks with a single fold, not the full ensemble model. Not recommended (default: False)

GPU detected
```

## Usage
Skellytour requires only a gzipped NIfTI file of a CT scan as input. The default usage is:
```
skellytour -i /path/to/input/nifti.nii.gz
```
This will run the "medium" model with 38 labels and write output to the current working directory. When a model is run for the first time, Skellytour will download the pretrained model from GitHub and store it in a hidden directory in the current user's home directory e.g. `/home/user/.skellytour`. Each model requires approximately 1.2 GB of storage space.

The following arguments are available:

Argument | Description
--- | ---
**`-h, --help`** | show this help message and exit
**`-i`** | path to input NIfTI file (default: None)
**`-o`** | path to output directory (default: .)
**`-m`** | model to use; can be low (17 labels), medium (38 labels, default), high (60 labels) (default: medium)
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
While running, Skellytour will print messages to the screen updating users on the process. Output files are gzipped NIfTI files containing multilabel segmentations. As an example, the command below will run the medium and subsegmentation models and write output to the `outputdir` directory:
```
skellytour -i example.nii.gz -o outputdir -m medium --subseg
```

Output file | Description
--- | ---
**log.txt** | Text file containing all details of the process
**example_medium.nii.gz** | Raw segmentation produced by the medium model
**example_medium_postprocessed.nii.gz** | Postprocessed medium model segmentation
**example_medium_postprocessed_subseg.nii.gz** |  Raw segmentation produced by the subsegmentation model
**example_medium_postprocessed_subseg_postprocessed.nii.gz** | Postprocessed subsegmentation model segmentation


## Getting Help
If you find an issue not covered in this document, please open a new issue on GitHub.

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
**High** | 36 | VERT_1
**High** | 37 | VERT_2
**High** | 38 | VERT_3
**High** | 39 | VERT_4
**High** | 40 | VERT_5
**High** | 41 | VERT_6
**High** | 42 | VERT_7
**High** | 43 | VERT_8
**High** | 44 | VERT_9
**High** | 45 | VERT_10
**High** | 46 | VERT_11
**High** | 47 | VERT_12
**High** | 48 | VERT_13
**High** | 49 | VERT_14
**High** | 50 | VERT_15
**High** | 51 | VERT_16
**High** | 52 | VERT_17
**High** | 53 | VERT_18
**High** | 54 | VERT_19
**High** | 55 | VERT_20
**High** | 56 | VERT_21
**High** | 57 | VERT_22
**High** | 58 | VERT_23
**High** | 59 | VERT_24
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
**Medium** | 14 | VERT_1
**Medium** | 15 | VERT_2
**Medium** | 16 | VERT_3
**Medium** | 17 | VERT_4
**Medium** | 18 | VERT_5
**Medium** | 19 | VERT_6
**Medium** | 20 | VERT_7
**Medium** | 21 | VERT_8
**Medium** | 22 | VERT_9
**Medium** | 23 | VERT_10
**Medium** | 24 | VERT_11
**Medium** | 25 | VERT_12
**Medium** | 26 | VERT_13
**Medium** | 27 | VERT_14
**Medium** | 28 | VERT_15
**Medium** | 29 | VERT_16
**Medium** | 30 | VERT_17
**Medium** | 31 | VERT_18
**Medium** | 32 | VERT_19
**Medium** | 33 | VERT_20
**Medium** | 34 | VERT_21
**Medium** | 35 | VERT_22
**Medium** | 36 | VERT_23
**Medium** | 37 | VERT_24
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
**Low** | 14 | CERVICAL_VERTS
**Low** | 15 | THORACIC_VERTS
**Low** | 16 | LUMBAR_VERTS
**Low** | 17 | ARTIFACTS
**Subsegmentation** | 1 | TRABECULAR BONE
**Subsegmentation** | 2 | CORTICAL BONE




