import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(current_path))
sys.path.append(ROOT_DIR)


from config.config import *
import tensorflow as tf

from tensorflow import keras
from keras.models import Model
from keras.layers import Dropout, Conv2D, MaxPool2D, Input, Conv2DTranspose, concatenate, BatchNormalization, UpSampling2D
from keras.optimizers import Adam
from keras.models import load_model
from keras.metrics import MeanIoU


from keras import backend as K
from sklearn.metrics import confusion_matrix

import tifffile as tiff

import numpy as np

import psutil

import matplotlib.pyplot as plt
import glob
import cv2




def mini_batch(count, patch_path, mask_path, path_len):

  imgs_input = []
  imgs_output = []

  if path_len - count >= FIX_QUANTITY:
    for i in range(count, count + FIX_QUANTITY):
        imgs_input.append(tiff.imread(f"{patch_path}/slice_patch_{i}.tiff"))
        imgs_output.append(tiff.imread(f"{mask_path}/mask_patch_{i}.tiff"))

    print(f"Successfully load from {count} to {count + FIX_QUANTITY}")

  else:
    for i in range(count, path_len):
        imgs_input.append(tiff.imread(f"{patch_path}/slice_patch_{i}.tiff"))
        imgs_output.append(tiff.imread(f"{mask_path}/mask_patch_{i}.tiff"))

    print(f"Successfully load from {count} to {path_len}")


  imgs_input = np.array(imgs_input, dtype=np.float32).reshape(-1, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS)
  imgs_output = np.array(imgs_output, dtype=np.float32).reshape(-1, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS)

  return imgs_input, imgs_output



if __name__=="__main__":

    model = load_model(SAVE_PRUNNING_MODEL_PATH, custom_objects={'dice_coef_loss': LOSS})

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS, metrics=[METRIC, MeanIoU(num_classes=2)])


    history_train = []
    history_validation = []
    history_test = []





    count = 0
    while count < LENGTH_TRAIN:
        imgs_input, imgs_output = mini_batch(count, PATH_PATCHES_TRAIN, PATH_MASKS_TRAIN, LENGTH_TRAIN)
        history_train.append(model.evaluate(imgs_input, imgs_output))

        count += FIX_QUANTITY


    count = 0
    while count < LENGTH_VALIDATION:
        imgs_input, imgs_output = mini_batch(count, PATH_PATCHES_VALIDATION, PATH_MASKS_VALIDATION, LENGTH_VALIDATION)
        history_validation.append(model.evaluate(imgs_input, imgs_output))

        count += FIX_QUANTITY


    count = 0
    while count < LENGTH_TEST:
        imgs_input, imgs_output = mini_batch(count, PATH_PATCHES_TEST, PATH_MASKS_TEST, LENGTH_TEST)
        history_test.append(model.evaluate(imgs_input, imgs_output))

        count += FIX_QUANTITY



    history_train = np.array(history_train)
    history_train = np.mean(history_train, axis=0)

    history_validation = np.array(history_validation)
    history_validation = np.mean(history_validation, axis=0)

    history_test = np.array(history_test)
    history_test = np.mean(history_test, axis=0)


    save_info = f'\nTrain: Dice coef: {history_train[1]} - Mean IoU: {history_train[2]}\nValidation: Dice coef: {history_validation[1]} - Mean IoU: {history_validation[2]}\nTest: Dice coef: {history_test[1]} - Mean IoU: {history_test[2]}'

    with open(EVALUATION_PATH, 'a') as f:
        f.write(save_info)