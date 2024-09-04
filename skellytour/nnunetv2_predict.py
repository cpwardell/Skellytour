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

from multiprocessing import Pool
import os
import torch
import pickle
import numpy as np
import contextlib
import sys

## DummyFile and nostdout() allow nnunet messages to be silenced
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

## Silently import nnUNetv2 predictor
with nostdout():
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
def predict_case(args,model_folder_name,folds,output_filenames,use_mirroring):

    ## Define compute device based on arguments
    if args.d == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
        perform_everything_on_device=False
    if args.d == 'gpu':
        device = torch.device('cuda',args.g)
        perform_everything_on_device=True
    if args.d == 'mps':
        device = torch.device('mps')
        perform_everything_on_device=False

    ## Set up predictor
    with nostdout():
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            #allow_tqdm=False # disable progress bar
            allow_tqdm=True # enable progress bar
        )
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_folder_name,
            use_folds=folds,
            checkpoint_name='checkpoint_final.pth',
        )
        predictor.predict_from_files(list_of_lists_or_source_folder=[[os.path.join(args.o,"temp.nii.gz")]],
                                 output_folder_or_list_of_truncated_output_files=output_filenames,
                                 save_probabilities=False, overwrite=args.overwrite,
                                 num_processes_preprocessing=args.c,
                                 num_processes_segmentation_export=args.c,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

