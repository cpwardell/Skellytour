#!/usr/bin/env python

# Copyright 2022-2023 CP Wardell
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Title: Skelly: Bone Segmentation from CT scans
## Author: CP Wardell
## Description: Creates bone segmentations from CT data

## TO DO
#Helpers to suppress stdout prints from nnunet
#https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto

##


## Import packages
import os
import sys
import argparse
import logging
import datetime
import torch

from skelly.nnunetv1_setup import nnunetv1_setup, nnunetv1_weights
from skelly.nnunetv1_predict import predict_case

def main():

    ## Gather command line args
    ## Create a new argparse class that will print the help message by default
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    parser=MyParser(description="Skelly: Bone Segmentation from CT scans", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", type=str, help="absolute path to input NIfTI file", required=True)
    parser.add_argument("-o", type=str, help="absolute path to output directory", required=True)
    parser.add_argument("-m", type=str, help="model to use", required=False, default="medium")
    args=parser.parse_args()

    ## Turn arguments into a nice string for printing
    printargs=str(sys.argv).replace(",","").replace("'","").replace("[","").replace("]","")

    ## Create directory to put results in
    ## Trycatch prevents exception if location is unwritable 
    try:
        os.makedirs(args.o,exist_ok=True)
    except Exception as e:
        logging.error("CRITICAL ERROR: results directory could not be created: "+str(e))
        logging.error("If using Docker, the current directory must be writeable by any user")
        sys.exit()

    ## Set up logging
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',filename=os.path.join(args.o,'log.txt'),filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    ## Write the command line parameters to the log file
    starttime=datetime.datetime.now()
    logging.info("Skelly was invoked using this command: "+printargs)
    logging.info("Start time is: "+str(starttime))
    logging.info("Input file is: "+str(args.i))
    logging.info("Output directory is: "+str(args.o))
    logging.info("Model used is: "+str(args.m))
    
    ## Check for GPUs
    gpustatus=torch.cuda.is_available()
    if gpustatus:
        computedevice="gpu"
        logging.info("Compute device is: "+str(computedevice))
    else:
        computedevice="cpu"
        logging.info("Compute device is: "+str(computedevice))
        logging.warning("No GPU detected, prediction will be much slower")



    ## Set up nnunet
    ## We need these three environment variables:
    #export nnUNet_raw_data_base="/data"
    #export nnUNet_preprocessed="/data/preprocessed"
    #export RESULTS_FOLDER="/data/results"
    nnunetdir=nnunetv1_setup()

    ## Check that weights exist; if not, go get them
    nnunetv1_weights(args.m,nnunetdir)

    ## Run Skelly
    ## Example nnUNet_predict; this calls predict_simple.py
    #nnUNet_predict -i /data/temp 
    # -o /data/OUTPUT/OUTPUT815_fullres_trimmed816 
    # -t 815 
    # -m 3d_fullres  
    # -tr nnUNetTrainerV2_noMirroring 
    # --disable_tta

    ## Check if the model weights exist.  If not, download them to a directory in the user's home dir e.g. 
    # .skelly/nnunet/results/nnUNet/3d_fullres/Task815_75/nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1
    # each zip file should contain everything from Taskxxx onwards

#    predict_case(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds,
#                  save_npz, num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations, tta,
#                  mixed_precision=mixed_precision, overwrite_existing=overwrite_existing,
#                  all_in_gpu=all_in_gpu,
#                  step_size=step_size, checkpoint_name=checkpoint_name,
#                  segmentation_export_kwargs=segmentation_export_kwargs,
#                  disable_postprocessing=disable_postprocessing)
#

    ## model variable
    #model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + args.plans_identifier)
    #os.path.join


## Execute main method
if __name__ == '__main__':
    main()

