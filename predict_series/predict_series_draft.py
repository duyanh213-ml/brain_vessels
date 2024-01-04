# -*- coding: utf-8 -*-

"""
===========================================================================================================

Created on Thu Aug 10 2023
Latest update: 08-10-23 

@author: AnhND
===========================================================================================================
"""




import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(current_path))
sys.path.append(ROOT_DIR)

from config.config import *
from tensorflow import keras

from keras import backend as K
import SimpleITK as sitk

import numpy as np

import multiprocessing as mp
import time
import nibabel as nib
import cv2

from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt




# This min-max normaliztion is applied after z-score
def min_max_normalization(series, max, min, epsilon=1e-9):
    # use epsilon to avoid underflow (divided by 0)
    return (series - min) / (max - min + epsilon)


def z_score_normalization(series, epsilon=1e-9):
    # The use of epsilon is similar to the min_max_normalization
    return (series - np.mean(series)) / (np.std(series) + epsilon)



def get_grid_points(width, height, patch_size):
    half_patch_size = patch_size // 2

    x_coords = np.arange(half_patch_size, width - half_patch_size + 1, patch_size)
    y_coords = np.arange(half_patch_size, height - half_patch_size + 1, patch_size)

    grid_points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    return grid_points



def pred_slice(slice, patch_size, model):
    width, height = slice.shape
    half_patch_size = patch_size // 2

    points = get_grid_points(width=width, height=height, patch_size=patch_size)

    num_patches = len(points)
    small_blocks = np.zeros((num_patches, patch_size, patch_size))

    for i, (x, y) in enumerate(points):
        small_block = slice[x - half_patch_size:x + half_patch_size, y - half_patch_size:y + half_patch_size]
        small_blocks[i] = small_block

    predictions = model.predict(small_blocks.reshape(num_patches, patch_size, patch_size, 1), verbose=0)
    predictions = np.where(predictions < 0.5, 0, 1)

    large_block = np.zeros((width, height), dtype=np.int8)  # Ensure dtype is specified

    for i, (x, y) in enumerate(points):
        large_block[x - half_patch_size:x + half_patch_size, y - half_patch_size:y + half_patch_size] = predictions[i, :, :, 0]

    return large_block


def pred_series(series, patch_size, model):
    slices_num = len(series)
    pred_series_save = []

    for i in range(slices_num):
        start = time.time()
        pred_series_save.append(pred_slice(slice=series[i], patch_size=patch_size, model=model))
        stop = time.time()
        
        print(f"Slice {i + 1}: Complete!!! {stop - start} s")

    return np.array(pred_series_save, dtype=np.int8)



# This is for multi-process
def pred_series_multiprocess(series, patch_size, model):
    slices_num = len(series)
    pred_series_save = []

    # Define the number of concurrent threads (adjust according to your CPU)
    num_threads = min(4, slices_num)  # You can increase or decrease this value

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(pred_slice, slice=series[i], patch_size=patch_size, model=model) for i in range(slices_num)]

        for future in futures:
            pred_series_save.append(future.result())
            print(f"Slice {len(pred_series_save)}: Complete!!!")

    return np.array(pred_series_save, dtype=np.int8)


path_segment = '/your/dir/file.nii'  # Should change the folder name first
path_dicom = '/your/dir/dicom_folder'  # Should change the folder name first

path_max_min = os.path.join(ROOT_DIR, '', 'processing_data/max_and_min.txt')


model = keras.models.load_model(SAVE_PRUNNING_MODEL_PATH, custom_objects={'dice_coef_loss': LOSS})


if __name__ == "__main__":
    # Get the slices and convert them to numpy array
    reader = sitk.ImageSeriesReader()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(path_dicom)
    reader.SetFileNames(filenamesDICOM)
    img_original = reader.Execute()
    img = sitk.GetArrayFromImage(img_original)

    # Read the mask image and convert them to numpy array
    mask = sitk.ReadImage(path_segment)
    mask = sitk.GetArrayFromImage(mask)

    #  Normalizing process:
    # 1. Z-score norm
    series = z_score_normalization(img)

    # Get the min-max value for min_max_normalization(series, max, min, epsilon=1e-7)
    with open(path_max_min, 'r') as f:
        lst = f.read().split(',')


    # To get the min, max value of training dataset
    min_value = float(lst[0])
    max_value = float(lst[1])



    # 2. Min-max norm
    series = min_max_normalization(series, max_value, min_value).astype(np.float32)



    print("Completely!! Our data has been normalized!!")
    
    imgs = pred_series(series=series, patch_size=96, model=model)


    kernel = np.ones((3, 4), np.uint8)

    imgs_pos = cv2.erode(imgs.astype(float), kernel=kernel, iterations=1)
    imgs_pos = cv2.dilate(imgs_pos, kernel=kernel, iterations=1)

    imgs_pos = imgs_pos.astype(np.int8)


    # Create a NIfTI image object
    nifti_img = nib.Nifti1Image(imgs, affine=np.eye(4))

    # Save the NIfTI image to a .nii file
    nib.save(nifti_img, './bn1.nii')
