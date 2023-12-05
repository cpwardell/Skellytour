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
    nnunetdir=os.path.join(os.path.expanduser(nnunetdir),".skellytour")
    logging.info("Models are stored here: "+str(nnunetdir))
    os.makedirs(nnunetdir, exist_ok=True)

    ## Export required nnunet environment variables (only :)
    os.environ['nnUNet_raw_data_base'] = nnunetdir
    os.environ['nnUNet_preprocessed'] = nnunetdir
    os.environ['RESULTS_FOLDER'] = os.path.join(nnunetdir,"results")

    ## Create the results folder and some subdirectories
    os.makedirs(os.path.join(os.environ['RESULTS_FOLDER'],"nnUNet/3d_fullres"), exist_ok=True)
    
    return(nnunetdir)


def nnunetv1_weights(model,nnunetdir):
    ## Check that the required model exists locally.  If not, fetch it.
    extant_models=["low","medium","high","subseg"]

    ## If model name is incorrect, exit
    if model not in extant_models:
        quoted_models=str(extant_models).replace("[","").replace("]","")
        logging.error(model+" is not an acceptable model name.  Please select from "+quoted_models)
        sys.exit()

    ## Set variables for each model, including where to find files (locally or online)
    if model=="low":
        taskname="Task812"
        taskno="812"
        fulltrainername="nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1"
        modelfolds=range(5)
        modelurl="https://github.com/cpwardell/Skellytour/releases/download/v0.0.1/Task812.zip"
    if model=="medium":
        taskname="Task815"
        taskno="815"
        fulltrainername="nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1"
        modelfolds=range(5)
        modelurl="https://github.com/cpwardell/Skellytour/releases/download/v0.0.1/Task815.zip"
    if model=="high":
        taskname="Task810"
        taskno="810"
        fulltrainername="nnUNetTrainerV2_noMirroring__nnUNetPlansv2.1"
        modelfolds=range(5)
        modelurl="https://github.com/cpwardell/Skellytour/releases/download/v0.0.1/Task810.zip"
    if model=="subseg":
        taskname="Task850"
        taskno="850"
        fulltrainername="nnUNetTrainerV2__nnUNetPlansv2.1"
        modelfolds=range(5)
        modelurl="https://github.com/cpwardell/Skellytour/releases/download/v0.0.1/Task850.zip"

    ## Check files exist; if not, fetch them
    models_and_pkls=[]
    for i in modelfolds:
        thisfold="fold_"+str(i)
        foldpath=os.path.join(os.environ['RESULTS_FOLDER'],"nnUNet/3d_fullres",taskname,fulltrainername,thisfold)
        thismodel=os.path.join(foldpath,"model_final_checkpoint.model")
        thispkl=thismodel+".pkl"
        models_and_pkls.append(os.path.isfile(thismodel))
        models_and_pkls.append(os.path.isfile(thispkl))

    if not all(models_and_pkls):
        ## Download file
        localzip=os.path.join(os.environ['RESULTS_FOLDER'],"nnUNet/3d_fullres/temp.zip")
        logging.info(model+" model not found locally, downloading zip file to "+localzip)
        r = requests.get(modelurl)
        with open(localzip, "wb") as f:
            f.write(r.content)
        ## Unzip file
        logging.info("Download complete, unzipping file")
        with zipfile.ZipFile(localzip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(os.environ['RESULTS_FOLDER'],"nnUNet/3d_fullres"))
        logging.info("Unzipping complete, removing temporary zip file")
        os.remove(localzip)

    ## Return some variables so we know where to find things
    return(taskname,taskno,fulltrainername)




