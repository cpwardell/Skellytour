#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    NOTICE: This code derived from nnunetv1: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1

import os
import logging
import sys
import requests
import zipfile

## By default, put things in user directory
def nnunetv1_setup(nnunetdir="~"):

    ## Create base directory to store all nnunet models
    nnunetdir=os.path.join(os.path.expanduser(nnunetdir),".skelly")
    logging.info("Models and temporary data stored here: "+str(nnunetdir))
    os.makedirs(nnunetdir, exist_ok=True)

    ## Export required nnunet environment variables (only :)
    os.environ['nnUNet_raw_data_base'] = nnunetdir
    os.environ['nnUNet_preprocessed'] = nnunetdir
    os.environ['RESULTS_FOLDER'] = os.path.join(nnunetdir,"results")

    ## Create the results folder
    os.makedirs(os.environ['RESULTS_FOLDER'], exist_ok=True)
    return(nnunetdir)


def nnunetv1_weights(model,nnunetdir):
    ## Check that the required model exists locally.  If not, fetch it.
    extant_models=["low","medium","high"]

    ## If model name is incorrect, exit
    if model not in extant_models:
        quoted_models=str(extant_models).replace("[","").replace("]","")
        logging.error(model+" is not an acceptable model name.  Please select from "+quoted_models)
        sys.exit()

    ## Set location to look for model files
    if model=="medium":
        taskname="Task666" ## should be 815
        taskno="666" ## should be 815
        trainername="nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1"
        modelfolds=range(5)
        #modelurl="https://github.com/cpwardell/Skelly/releases/download/untagged-e81e4403ed82722a11f5/Task815.zip"
        modelurl="https://github.com/cpwardell/Skelly/releases/download/untagged-e81e4403ed82722a11f5/Task666.zip"

    ## Check files exist; if not, fetch them
    #.skelly/nnunet/results/nnUNet/3d_fullres/Task815_75/nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1
    models_and_pkls=[]
    for i in modelfolds:
        thisfold="fold_"+str(i)
        foldpath=os.path.join(nnunetdir,"nnunet/results/nnUNet/3d_fullres",taskname,trainername,thisfold)
        thismodel=os.path.join(foldpath,"model_final_checkpoint.model")
        thispkl=thismodel+".pkl"
        models_and_pkls.append(os.path.isfile(thismodel))
        models_and_pkls.append(os.path.isfile(thispkl))

    if not all(models_and_pkls):
        ## Download file
        localzip=os.path.join(os.environ['RESULTS_FOLDER'],"temp.zip")
        logging.info(model+" model not found locally, downloading zip file to "+localzip)
        r = requests.get(modelurl)
        with open(localzip, "wb") as f:
            f.write(r.content)
        sys.exit()
        ## Unzip file
        logging.info("Download complete, unzipping file")
        with zipfile.ZipFile(localzip, 'r') as zip_ref:
            zip_ref.extractall(os.environ['RESULTS_FOLDER'])
        logging.info("Unzipping complete")

#def is_zip_file(file_path):
#    try:
#        with zipfile.ZipFile(file_path) as zf:
#            return zf.testzip() is None
#    except zipfile.BadZipFile:
#        return False

        #modelname="3d_fullres"
        # -tr nnUNetTrainerV2_noMirroring 
        # --disable_tta

    #file_path = "/path/to/file.txt"
    #if os.path.isfile(file_path):
    #    print("File exists")
    #else:
    #    print("File does not exist")


