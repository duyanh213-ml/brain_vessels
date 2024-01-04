# -*- coding: utf-8 -*-

"""
===========================================================================================================

Created on Thu Jun 29 2023
Latest update: 07-13-23 

@author: AnhND
===========================================================================================================
"""


from process_data_config import *


import generate_patches
import normalizing_data


def pipeline(patch_size, path_dicom, path_segment, path_patches, path_mask, flag=False):
    # Normalizing data
    normalized_data = normalizing_data.normalized_process(
        path_dicom=path_dicom, path_segment=path_segment, flag=flag)

    # Get the series list and mask list from normalizing data
    series_list = normalized_data['series']
    mask_list = normalized_data['masks']

    # Generate non-artery-patches and responding mask

    # Get the idx-non-artery to keep gen the index for artery
    idx_nonartery, step_size = generate_patches.generate_non_artery_patches(
        series_list=series_list, mask_list=mask_list, patch_size=patch_size,
        path_patches=path_patches, path_mask=path_mask, flag=flag)
    
    if flag:
        # Generate artery-patches and responding mask
        generate_patches.generate_artery_patches(
            series_list=series_list, mask_list=mask_list, patch_size=patch_size,
            path_patches=path_patches, path_mask=path_mask, idx_nonartery=idx_nonartery, step_size=step_size)



def run_total_pipeline():

    # Specify the path 
    patch_size = PATCH_SIZE


    # Specify the paths for patches
    path_patches_train_train = PATH_PATCHES_TRAINING_TRAINING
    path_mask_train_train = PATH_MASK_TRAINING_TRAINING

    path_patches_train_eval = PATH_PATCHES_TRAINING_EVAL
    path_mask_train_eval = PATH_MASK_TRAINING_EVAL

    path_patches_validation = PATH_PATCHES_VALIDATION
    path_mask_validation = PATH_MASK_VALIDATION

    path_patches_test = PATH_PATCHES_TEST
    path_mask_test = PATH_MASK_TEST

    # Specify the paths for dicom and masks
    path_dicom_training = PATH_DICOM_TRAINING
    path_segment_training = PATH_SEGMENT_TRAINING

    path_dicom_validation = PATH_DICOM_VALIDATION
    path_segment_validation = PATH_SEGMENT_VALIDATION

    path_dicom_test = PATH_DICOM_TEST
    path_segment_test = PATH_SEGMENT_TEST


    # Running pipeline:

    # -----------------------------Training - Training ------------------------------------------------------
    pipeline(patch_size=patch_size, path_dicom=path_dicom_training, path_segment=path_segment_training,
             path_patches=path_patches_train_train, path_mask=path_mask_train_train, flag=True)
    
    print("Completing the data pre-processing pipeline for training-training!!!")
    
    
    # -----------------------------Training - Evaluation ------------------------------------------------------
    pipeline(patch_size=patch_size, path_dicom=path_dicom_training, path_segment=path_segment_training,
             path_patches=path_patches_train_eval, path_mask=path_mask_train_eval)
    
    print("Completing the data pre-processing pipeline for training-evaluating!!!")

    # ----------------------------- Validation ------------------------------------------------------
    
    pipeline(patch_size=patch_size, path_dicom=path_dicom_validation, path_segment=path_segment_validation,
             path_patches=path_patches_validation, path_mask=path_mask_validation)
    
    print("Completing the data pre-processing pipeline for validation!!!")
    
    # ----------------------------- Test ------------------------------------------------------
    
    pipeline(patch_size=patch_size, path_dicom=path_dicom_test, path_segment=path_segment_test,
             path_patches=path_patches_test, path_mask=path_mask_test)
    
    print("Completing the data pre-processing pipeline for test!!!")
   
    # Successfull notification
    print("Completing the data pre-processing pipeline!!!")


if __name__ == "__main__":
    run_total_pipeline()