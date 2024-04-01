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
#    NOTICE: This code derived from nnunetv2: https://github.com/MIC-DKFZ/nnUNet

import os
import logging
import sys
import requests
import zipfile

## By default, put things in user directory
def nnunetv2_setup(nnunetdir="~"):

    ## Create base directory to store all nnunet models
    nnunetdir=os.path.join(os.path.expanduser(nnunetdir),".skellytour")
    logging.info("Models are stored here: "+str(nnunetdir))
    os.makedirs(nnunetdir, exist_ok=True)

    ## Export required nnunet environment variables (only nnUNet_results is used by Skellytour)
    os.environ['nnUNet_raw'] = nnunetdir
    os.environ['nnUNet_preprocessed'] = nnunetdir
    os.environ['nnUNet_results'] = os.path.join(nnunetdir,"nnUNet_results")

    ## Create the results folder to store models in
    os.makedirs(os.path.join(os.environ['nnUNet_results']), exist_ok=True)
    
    return(nnunetdir)


def nnunetv2_weights(model,nnunetdir):
    ## Set variables for each model, including where to find files (locally or online)
    if model=="low":
        model_folder_name=os.path.join(os.environ['nnUNet_results'],"Dataset812_skellylow/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres")
        use_mirroring=False
        modelfolds=range(5)
        modelurl="https://github.com/cpwardell/Skellytour/releases/download/v0.0.2/Dataset812.zip"
    if model=="medium":
        model_folder_name=os.path.join(os.environ['nnUNet_results'],"Dataset815_skelly75/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres")
        use_mirroring=False
        modelfolds=range(5)
        modelurl="https://github.com/cpwardell/Skellytour/releases/download/v0.0.2/Dataset815.zip"
    if model=="high":
        model_folder_name=os.path.join(os.environ['nnUNet_results'],"Dataset810_skellyhigh/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres")
        use_mirroring=False
        modelfolds=range(5)
        modelurl="https://github.com/cpwardell/Skellytour/releases/download/v0.0.2/Dataset810.zip"
    if model=="subseg":
        model_folder_name=os.path.join(os.environ['nnUNet_results'],"Dataset850_skellsubseg/nnUNetTrainer__nnUNetPlans__3d_fullres")
        use_mirroring=True
        modelfolds=range(5)
        modelurl="https://github.com/cpwardell/Skellytour/releases/download/v0.0.2/Dataset850.zip"


    ## Check files exist; if not, fetch them
    checkpoints=[]
    for i in modelfolds:
        thisfold="fold_"+str(i)
        foldpath=os.path.join(model_folder_name,thisfold)
        thismodel=os.path.join(foldpath,"checkpoint_final.pth")
        checkpoints.append(os.path.isfile(thismodel))

    if not all(checkpoints):
        ## Download file
        localzip=os.path.join(os.environ['nnUNet_results'],"temp.zip")
        logging.info(model+" model not found locally, downloading zip file to "+localzip)
        r = requests.get(modelurl)
        with open(localzip, "wb") as f:
            f.write(r.content)
        ## Unzip file
        logging.info("Download complete, unzipping file")
        with zipfile.ZipFile(localzip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(os.environ['nnUNet_results']))
        logging.info("Unzipping complete, removing temporary zip file")
        os.remove(localzip)

    ## Return some variables so we know where to find things
    return(model_folder_name,use_mirroring)

