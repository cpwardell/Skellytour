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

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

with nostdout():
    from nnunet.training.model_restore import load_model_and_checkpoint_files
    from nnunet.inference.predict import preprocess_multithreaded
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax

def predict_case(model, list_of_lists, output_filenames, folds,save_npz, num_threads_preprocessing,
                 num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True,
                 overwrite_existing=False,
                 all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                 segmentation_export_kwargs: dict = None, disable_postprocessing: bool = False):
    """
    :param segmentation_export_kwargs:
    :param model: folder where the model is saved, must contain fold_x subfolders
    :param list_of_lists: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
    :param output_filenames: [output_file_case0.nii.gz, output_file_case1.nii.gz, ...]
    :param folds: default: (0, 1, 2, 3, 4) (but can also be 'all' or a subset of the five folds, for example use (0, )
    for using only fold_0
    :param save_npz: default: False
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param segs_from_prev_stage:
    :param do_tta: default: True, can be set to False for a 8x speedup at the cost of a reduced segmentation quality
    :param overwrite_existing: default: True
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """

    with nostdout():
        assert len(list_of_lists) == len(output_filenames)
        if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

        pool = Pool(num_threads_nifti_save)
        results = []

        cleaned_output_files = []
        for o in output_filenames:
            dr, f = os.path.split(o)
            #if len(dr) > 0:
            #    maybe_mkdir_p(dr)
            if not f.endswith(".nii.gz"):
                f, _ = os.path.splitext(f)
                f = f + ".nii.gz"
            cleaned_output_files.append(os.path.join(dr, f))

        if not overwrite_existing:
            #print("number of cases:", len(list_of_lists))
            # if save_npz=True then we should also check for missing npz files
            not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not os.path.isfile(j)) or (save_npz and not os.path.isfile(j[:-7] + '.npz'))]
           #not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not os.path.isfile(j)) or (not isfile(j[:-7] + '.npz'))]

            cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
            list_of_lists = [list_of_lists[i] for i in not_done_idx]
            if segs_from_prev_stage is not None:
                segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

            #print("number of cases that still need to be predicted:", len(cleaned_output_files))

        #print("emptying cuda cache")
        torch.cuda.empty_cache()

        #print("loading parameters for folds,", folds)
        trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                          checkpoint_name=checkpoint_name)

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in trainer.plans.keys():
                force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        #print("starting preprocessing generator")
        preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                                 segs_from_prev_stage)
        #print("starting prediction...")
        all_output_files = []
        for preprocessed in preprocessing:
            output_filename, (d, dct) = preprocessed
            all_output_files.append(all_output_files)
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

            #print("predicting", output_filename)
            trainer.load_checkpoint_ram(params[0], False)
            softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1]

            for p in params[1:]:
                trainer.load_checkpoint_ram(p, False)
                softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                    d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                    step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                    mixed_precision=mixed_precision)[1]

            if len(params) > 1:
                softmax /= len(params)

            transpose_forward = trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = trainer.plans.get('transpose_backward')
                softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

            #if save_npz:
            #    npz_file = output_filename[:-7] + ".npz"
            #else:
            npz_file = None

            if hasattr(trainer, 'regions_class_order'):
                region_class_order = trainer.regions_class_order
            else:
                region_class_order = None

            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            bytes_per_voxel = 4
            if all_in_gpu:
                bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
            if np.prod(softmax.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
                #print(
                #    "This output is too large for python process-process communication. Saving output temporarily to disk")
                np.save(output_filename[:-7] + ".npy", softmax)
                softmax = output_filename[:-7] + ".npy"
            # save_segmentation_nifti_from_softmax(softmax, output_filename, dct, interpolation_order, region_class_order,
            #                                     None, None,
            #                                     npz_file, None, force_separate_z, interpolation_order_z)

            results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                              ((softmax, output_filename, dct, interpolation_order, region_class_order,
                                                None, None,
                                                npz_file, None, force_separate_z, interpolation_order_z),)
                                              ))

        #print("inference done. Now waiting for the segmentation export to finish...")
        _ = [i.get() for i in results]
        # now apply postprocessing
        # first load the postprocessing properties if they are present. Else raise a well visible warning
        if not disable_postprocessing:
            results = []
            pp_file = os.path.join(model, "postprocessing.json")
            if os.path.isfile(pp_file):
                #print("postprocessing...")
                shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
                # for_which_classes stores for which of the classes everything but the largest connected component needs to be
                # removed
                for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
                results.append(pool.starmap_async(load_remove_save,
                                                  zip(output_filenames, output_filenames,
                                                      [for_which_classes] * len(output_filenames),
                                                      [min_valid_obj_size] * len(output_filenames))))
                _ = [i.get() for i in results]
            #else:
            #    print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
            #          "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
            #          "%s" % model)

        pool.close()
        pool.join()

