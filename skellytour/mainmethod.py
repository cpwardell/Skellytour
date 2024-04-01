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
import cpuinfo

from skellytour.nnunetv2_setup import nnunetv2_setup, nnunetv2_weights
from skellytour.nnunetv2_predict import predict_case 
from skellytour.postprocessing import postprocessing
from skellytour.subseg_postprocessing import subsegpostprocessing

def exitlog(starttime):
    endtime=datetime.datetime.now()
    logging.info("End time is: "+str(endtime.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info("Total time taken: "+str(endtime-starttime))
    sys.exit()

def filedelete(pathtofile):
    if os.path.exists(pathtofile):
        try:
            os.remove(pathtofile)
        except:
            logging.error("Could not delete file: "+str(pathtofile))

def main():

    ## Check if GPU is available and print count
    gpustatus=torch.cuda.is_available()
    if gpustatus:
        devicecount=torch.cuda.device_count()
        if(devicecount==1):
            gputext="GPU"
        else:
            gputext="GPUs"
        epilogtext=str(devicecount)+" "+gputext+" detected"
    else:
        epilogtext="No GPUs detected, prediction will be much slower"

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
    parser.add_argument("-m", type=str, help="model to use; can be low (17 labels), medium (38 labels, default), high (60 labels)", required=False, default="medium", choices=["low","medium","high"])
    parser.add_argument("-c", type=int, help="number of CPU cores to use for preprocessing and postprocessing", required=False, default=6)
    parser.add_argument("-d", type=str, help="compute device to use", required=False, default="gpu", choices=["gpu","cpu","mps"])
    parser.add_argument("-g", type=int, help="GPU to use", required=False, default=0)
    parser.add_argument("--overwrite", help="overwrite previous results if they exist", required=False, default=False, action='store_true')
    parser.add_argument("--nopp", help="skip postprocessing on predicted segmentations", required=False, default=False, action='store_true')
    parser.add_argument("--subseg", help="perform subsegmentation, to predict trabecular and cortical labels", required=False, default=False, action='store_true')
    parser.add_argument("--fast", help="perform segmentation tasks with a single fold, not the full ensemble model. Not recommended", required=False, default=False, action='store_true')
    args=parser.parse_args()

    ## Turn arguments into a nice string for printing
    printargs=str(sys.argv).replace(",","").replace("'","").replace("[","").replace("]","")

    ## Create directory to put results in, exit if location is unwritable 
    try:
        os.makedirs(args.o,exist_ok=True)
    except Exception as e:
        print("CRITICAL ERROR: results directory could not be created: "+str(e))
        print("If using Docker, the current directory must be writeable by any user")
        sys.exit()

    ## Avoid overwriting if output exists; check for a logfile
    if not args.overwrite and os.path.exists(os.path.join(args.o,'log.txt')):
        print("Log file found in output directory: "+os.path.join(args.o,'log.txt'))
        print("To overwrite existing output, append the --overwrite flag to your command")
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
    logging.info("If you use this software, please cite our upcoming paper:\nMann D.C., Rutherford M., Farmer P., Eichhorn J., Palot Manzil F.F., Wardell C.P. (2024).\nSkellytour: Automated Skeleton Segmentation from Whole-Body CT Images")
    logging.info("Skellytour was invoked using this command: "+printargs)
    logging.info("Start time is: "+str(starttime.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info("Input file is: "+str(args.i))
    logging.info("Output directory is: "+str(args.o))
    logging.info("Model used is: "+str(args.m))
    logging.info("CPU cores used for pre/postprocessing: "+str(args.c))

    ## Determine which compute device to use for prediction
    if args.d=="gpu":
        try:
            gpuname=torch.cuda.get_device_name(args.g)
            # Multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception as e:
            logging.error("CRITICAL ERROR: GPU "+str(args.g)+" is not available or does not exist: "+str(e))
            sys.exit()
        logging.info("Compute device is GPU "+str(args.g)+": "+gpuname)
    if args.d=="mps":
        if torch.backends.mps.is_available():
            logging.info("Compute device is MPS")
        else:
            logging.error("CRITICAL ERROR: MPS is not available")
            sys.exit()
    if args.d=="cpu":
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info['brand_raw']
        logging.info("Compute device is CPU: "+cpu_name)
        logging.warning("Compute device is CPU, prediction will be much slower")

    ## Set up nnunet
    nnunetdir=nnunetv2_setup()

    ## Check that weights exist; if not, go get them
    model_folder_name,use_mirroring=nnunetv2_weights(args.m,nnunetdir)

    ## Set up input variables for main prediction
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
        subsegmodel_folder_name,sugbseguse_mirroring=nnunetv2_weights("subseg",nnunetdir)
        subseg_filename=segmentation_filename[:-7]+"_postprocessed_subseg.nii.gz"
        subseg_postprocessed_filename=subseg_filename[:-7]+"_postprocessed.nii.gz"

    ## Avoid overwriting if output exists
    if not args.overwrite and os.path.exists(segmentation_filename):
        logging.info("Segmentation output already exists: "+str(segmentation_filename))
        logging.info("To overwrite existing output, append the --overwrite flag to your command")
    else:
        ## Do prediction
        logging.info("Prediction starting")
        predict_case(args=args,model_folder_name=model_folder_name,folds=folds,
            output_filenames=[segmentation_filename],use_mirroring=use_mirroring)
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
            predict_case(args=args,model_folder_name=subsegmodel_folder_name,folds=folds,
                output_filenames=[subseg_filename],use_mirroring=sugbseguse_mirroring)
            logging.info("Subsegmentation complete, output is: "+str(subseg_filename))
            logging.info("Performing subsegmentation postprocessing")
            subsegpostprocessing(subseg_filename,postprocessed_filename,subseg_postprocessed_filename)
            logging.info("Subsegmentation postprocessing complete, output is: "+str(subseg_postprocessed_filename))

    ## Remove json file clutter
    logging.info("Removing unnecessary json files")
    filedelete(os.path.join(args.o,"dataset.json"))
    filedelete(os.path.join(args.o,"plans.json"))
    filedelete(os.path.join(args.o,"predict_from_raw_data_args.json"))

    ## Wrap up
    exitlog(starttime)


## Execute main method
if __name__ == '__main__':
    main()

