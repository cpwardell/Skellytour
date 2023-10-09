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

#from typing import Tuple, Union, List
#from typing import Union
from multiprocessing import Pool
import os
import torch
import pickle

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    search_in = os.path.join(nnunet.__path__[0], "training", "network_training")
    tr = recursive_find_python_class([search_in], name, current_module="nnunet.training.network_training")

    if tr is None:
        """
        Fabian only. This will trigger searching for trainer classes in other repositories as well
        """
        try:
            import meddec
            search_in = os.path.join(meddec.__path__[0], "model_training")
            tr = recursive_find_python_class([search_in], name, current_module="meddec.model_training")
        except ImportError:
            pass

    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    assert issubclass(tr, nnUNetTrainer), "The network trainer was found but is not a subclass of nnUNetTrainer. " \
                                          "Please make it so!"

    # this is now deprecated
    """if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of nnUNet. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]"""

    # ToDo Fabian make saves use kwargs, please...

    trainer = tr(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer

def load_model_and_checkpoint_files(folder, folds=None, mixed_precision=None, checkpoint_name="model_best"):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them from disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    if isinstance(folds, str):
        folds = [os.path.join(folder, "all")]
        assert os.path.isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [os.path.join(folder, "all")]
        else:
            folds = [os.path.join(folder, "fold_%d" % i) for i in folds]
        assert all([os.path.isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [os.path.join(folder, "fold_%d" % folds)]
        assert all([os.path.isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))

    trainer = restore_model(os.path.join(folds[0], "%s.model.pkl" % checkpoint_name), fp16=mixed_precision)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    trainer.initialize(False)
    all_best_model_files = [os.path.join(i, "%s.model" % checkpoint_name) for i in folds]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params

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
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(os.path.join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        # if save_npz=True then we should also check for missing npz files
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not os.path.isfile(j)) or (save_npz and not os.path.isfile(j[:-7] + '.npz'))]
       #not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not os.path.isfile(j)) or (not isfile(j[:-7] + '.npz'))]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
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

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)
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
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk")
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

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]
    # now apply postprocessing
    # first load the postprocessing properties if they are present. Else raise a well visible warning
    if not disable_postprocessing:
        results = []
        pp_file = os.path.join(model, "postprocessing.json")
        if os.path.isfile(pp_file):
            print("postprocessing...")
            shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
            # for_which_classes stores for which of the classes everything but the largest connected component needs to be
            # removed
            for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
            results.append(pool.starmap_async(load_remove_save,
                                              zip(output_filenames, output_filenames,
                                                  [for_which_classes] * len(output_filenames),
                                                  [min_valid_obj_size] * len(output_filenames))))
            _ = [i.get() for i in results]
        else:
            print("WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                  "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                  "%s" % model)

    pool.close()
    pool.join()

