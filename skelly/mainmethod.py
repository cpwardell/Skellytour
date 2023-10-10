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

## To do:
## Add other models
## Add postprocessing
## Add inner/outer method
#####


## Import packages
import os
import sys
import argparse
import logging
import datetime
import torch

from skelly.nnunetv1_setup import nnunetv1_setup, nnunetv1_weights
from skelly.nnunetv1_predict import predict_case 

def exitlog(starttime):
    endtime=datetime.datetime.now()
    logging.info("End time is: "+str(endtime.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info("Total time taken: "+str(endtime-starttime))
    sys.exit()

def main():

    ## Gather command line args
    ## Create a new argparse class that will print the help message by default
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    parser=MyParser(description="Skelly: Bone Segmentation from CT scans", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", type=str, help="path to input NIfTI file", required=True)
    parser.add_argument("-o", type=str, help="path to output directory", required=False, default=".")
    parser.add_argument("-m", type=str, help="model to use; can be low (17 labels), medium (38 labels, default), high (60 labels)", required=False, default="medium")
    parser.add_argument("--overwrite", help="overwrite previous results if they exist", required=False, default=False, action='store_true')
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
    logging.info("Start time is: "+str(starttime.strftime("%Y-%m-%d %H:%M:%S")))
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
    nnunetdir=nnunetv1_setup()

    ## Check that weights exist; if not, go get them
    taskname,taskno,fulltrainername=nnunetv1_weights(args.m,nnunetdir)

    ## Run Skelly
    ## Set up input variables for prediction    
    model_folder_name=os.path.join(os.environ['RESULTS_FOLDER'],"nnUNet/3d_fullres",taskname,fulltrainername)
    output_filename=os.path.join(args.o,os.path.basename(args.i))

    if not args.overwrite and os.path.exists(output_filename):
        logging.info("Output already exists: "+str(output_filename))
        logging.info("To overwrite existing output, append the --overwrite flag to your command")
        exitlog(starttime)

    logging.info("Prediction starting")
    predict_case(model=model_folder_name, list_of_lists=[[args.i]], output_filenames=[output_filename], folds=(0, 1, 2, 3, 4),save_npz=False,
                  num_threads_preprocessing=2, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=False,
                  mixed_precision=None, overwrite_existing=args.overwrite,
                  all_in_gpu=False,
                  step_size=0.5, checkpoint_name="model_final_checkpoint",
                  segmentation_export_kwargs=None,
                  disable_postprocessing=True)

    logging.info("Prediction complete, output is: "+str(output_filename))
    exitlog(starttime)


## Execute main method
if __name__ == '__main__':
    main()

