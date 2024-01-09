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

## Title: Skellytour: Bone Segmentation from CT scans
## Authors: CP Wardell and D Mann
## Description: Creates bone segmentations from CT data

## Import packages
import os
import sys
import argparse
import logging
import datetime
import torch

from skellytour.nnunetv1_setup import nnunetv1_setup, nnunetv1_weights
from skellytour.nnunetv1_predict import predict_case 
from skellytour.postprocessing import postprocessing
from skellytour.subseg_postprocessing import subsegpostprocessing

def exitlog(starttime):
    endtime=datetime.datetime.now()
    logging.info("End time is: "+str(endtime.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info("Total time taken: "+str(endtime-starttime))
    sys.exit()

def main():

    ## Check if GPU is available
    gpustatus=torch.cuda.is_available()
    if gpustatus:
        epilogtext="GPU detected"
    else:
        epilogtext="No GPU detected, prediction will be much slower"

    ## Gather command line args
    ## Create a new argparse class that will print the help message by default
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    parser=MyParser(description="Skellytour: Bone Segmentation from CT scans", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilogtext)
    parser.add_argument("-i", type=str, help="path to input NIfTI file", required=True)
    parser.add_argument("-o", type=str, help="path to output directory", required=False, default=".")
    parser.add_argument("-m", type=str, help="model to use; can be low (17 labels), medium (38 labels, default), high (60 labels)", required=False, default="medium")
    parser.add_argument("--overwrite", help="overwrite previous results if they exist", required=False, default=False, action='store_true')
    parser.add_argument("--nopp", help="skip postprocessing on predicted segmentations", required=False, default=False, action='store_true')
    parser.add_argument("--subseg", help="perform subsegmentation, assigning trabecular and cortical labels", required=False, default=False, action='store_true')
    parser.add_argument("--fast", help="perform segmentation tasks with a single fold, not the full ensemble model. Not recommended", required=False, default=False, action='store_true')
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
    logging.info("Skellytour was invoked using this command: "+printargs)
    logging.info("Start time is: "+str(starttime.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info("Input file is: "+str(args.i))
    logging.info("Output directory is: "+str(args.o))
    logging.info("Model used is: "+str(args.m))

    ## Print appropriate message regarding GPU status
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

    ## Set up input variables for main prediction    
    model_folder_name=os.path.join(os.environ['RESULTS_FOLDER'],"nnUNet/3d_fullres",taskname,fulltrainername)
    samplename=os.path.basename(args.i)[:-7]
    segmentation_filename=os.path.join(args.o,samplename+"_"+args.m+".nii.gz")
    postprocessed_filename=segmentation_filename[:-7]+"_postprocessed.nii.gz"

    ## Set up folds; if --fast is used, use only the 0th fold
    if args.fast:
        logging.warning("Fast mode enabled, only using a single fold for prediction")
        folds=(0,)
    else:
        folds=(0,1,2,3,4)

    ## Get model and set up input variables for subsegmentation
    if args.subseg:
        subsegtaskname,subsegtaskno,subsegfulltrainername=nnunetv1_weights("subseg",nnunetdir)
        subsegmodel_folder_name=os.path.join(os.environ['RESULTS_FOLDER'],"nnUNet/3d_fullres",subsegtaskname,subsegfulltrainername)
        subseg_filename=segmentation_filename[:-7]+"_postprocessed_subseg.nii.gz"
        subseg_postprocessed_filename=subseg_filename[:-7]+"_postprocessed.nii.gz"

    ## Avoid overwriting if output exists
    if not args.overwrite and os.path.exists(segmentation_filename):
        logging.info("Segmentation output already exists: "+str(segmentation_filename))
        logging.info("To overwrite existing output, append the --overwrite flag to your command")
    else:
        ## Do prediction
        logging.info("Prediction starting")
        predict_case(model=model_folder_name, list_of_lists=[[args.i]], output_filenames=[segmentation_filename], folds=folds,save_npz=False,
                      num_threads_preprocessing=2, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=False,
                      mixed_precision=None, overwrite_existing=args.overwrite,
                      all_in_gpu=False,
                      step_size=0.5, checkpoint_name="model_final_checkpoint",
                      segmentation_export_kwargs=None,
                      disable_postprocessing=True)
        logging.info("Prediction complete, output is: "+str(segmentation_filename))

    ## Perform postprocessing if desired and segmentation completed
    if not args.nopp and os.path.exists(segmentation_filename):
        ## Avoid overwriting if output exists
        if not args.overwrite and os.path.exists(postprocessed_filename):
            logging.info("Postprocessed output already exists: "+str(postprocessed_filename))
            logging.info("To overwrite existing output, append the --overwrite flag to your command")
        else:
            logging.info("Performing postprocessing")
            postprocessing(args)
            logging.info("Postprocessing complete, output is: "+str(postprocessed_filename))

    ## Perform subsegmentation and produce cortical/trabecular labels
    ## We only allow postprocessed segmentations as input
    if args.subseg and os.path.exists(postprocessed_filename):
        if not args.overwrite and os.path.exists(subseg_filename):
            logging.info("Subsegmentation output already exists: "+str(subseg_filename))
            logging.info("To overwrite existing output, append the --overwrite flag to your command")
        else:
            logging.info("Performing subsegmentation")
            predict_case(model=subsegmodel_folder_name, list_of_lists=[[args.i]], output_filenames=[subseg_filename], folds=folds,save_npz=False,
                      num_threads_preprocessing=2, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=False,
                      mixed_precision=None, overwrite_existing=args.overwrite,
                      all_in_gpu=False,
                      step_size=0.5, checkpoint_name="model_final_checkpoint",
                      segmentation_export_kwargs=None,
                      disable_postprocessing=True)
            logging.info("Subsegmentation complete, output is: "+str(subseg_filename))
            logging.info("Performing subsegmentation postprocessing")
            subsegpostprocessing(subseg_filename,postprocessed_filename,subseg_postprocessed_filename)
            logging.info("Subsegmentation postprocessing complete, output is: "+str(subseg_postprocessed_filename))

    ## Wrap up
    exitlog(starttime)


## Execute main method
if __name__ == '__main__':
    main()

