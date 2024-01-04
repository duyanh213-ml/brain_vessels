# -*- coding: utf-8 -*-

"""
===========================================================================================================

Created on Wed Jul 05 2023
Latest update: 07-13-23 

@author: AnhND
===========================================================================================================
"""
import os


# Path of min-max (for normalization)
PATH_MIN_MAX = './max_and_min.txt'

# Modify the patch size in here
PATCH_SIZE = 96


# These lines help us do not need to modify the directory path
current_path = os.path.dirname(os.path.abspath(__file__))

path_raw = os.path.join(current_path, '..', 'raw_data')


PATH_PATCHES = os.path.join(current_path, '..', 'patches')
PATH_MASK = os.path.join(current_path, '..', 'masks')



# Specify the paths to the processed data

PATH_PATCHES_TRAINING_TRAINING = f"{PATH_PATCHES}/patches_train_train/"
PATH_MASK_TRAINING_TRAINING = f"{PATH_MASK}/masks_train_train/"


PATH_PATCHES_TRAINING_EVAL = f"{PATH_PATCHES}/patches_train_eval/"
PATH_MASK_TRAINING_EVAL = f"{PATH_MASK}/masks_train_eval/"



PATH_PATCHES_VALIDATION = f"{PATH_PATCHES}/patches_validation/"
PATH_MASK_VALIDATION = f"{PATH_MASK}/masks_validation/"


PATH_PATCHES_TEST = f"{PATH_PATCHES}/patches_test/"
PATH_MASK_TEST = f"{PATH_MASK}/masks_test/"


# Specify the paths to the dicom and segmentation in raw data
PATH_DICOM_TRAINING = f'{path_raw}/train/train_dicom/'
PATH_SEGMENT_TRAINING = f'{path_raw}/train/train_segmentation_nii/'


PATH_DICOM_VALIDATION = f'{path_raw}/validation/validation_dicom/'
PATH_SEGMENT_VALIDATION = f'{path_raw}/validation/validation_segmentation_nii/'


PATH_DICOM_TEST = f'{path_raw}/test/test_dicom/'
PATH_SEGMENT_TEST = f'{path_raw}/test/test_segmentation_nii/'


print(PATH_DICOM_TEST)