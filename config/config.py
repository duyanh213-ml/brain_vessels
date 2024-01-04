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


import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from model.loss import dice_coef_loss, dice_coef






PATCH_SIZE = 96
NUM_CHANNELS = 1

LEARNING_RATE = 1e-6
DROPOUT = 0.1
LOSS = dice_coef_loss
METRIC = dice_coef

FIX_QUANTITY = 1024

BATCH_SIZE = 128



# These lines help us do not need to modify the directory path
current_path = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(current_path))



TOTAL_PATCHES_NUM = len(os.listdir(f"{ROOT_DIR}/masks/masks_train_train"))


INPUT_SHAPE = (96, 96, 1)
FILTERS_LIST = [64, 128, 256, 256, 512]
KERNEL_SIZE = (3, 3)
DOWN_STRIDES = (1, 1)
UP_STRIDES = (2, 2)
PADDING = "same"
ACTIVATION = "relu"
FINAL_ACTIVATION = "sigmoid"
KERNEL_INITIALIZER = "he_normal"
RATE = 0.1


LOSSES_SAVE_PATH = os.path.join(ROOT_DIR, '', 'loss_history/loss_history.txt')
SAVE_MODEL_PATH = os.path.join(ROOT_DIR, '', 'model_save/model.h5')


SAVE_PRUNNING_MODEL_PATH = os.path.join(ROOT_DIR, '', 'model_save/prunning_model.h5')

OPTIMIZER = Adam(learning_rate=LEARNING_RATE)

STEPS_PER_EPOCH = int(TOTAL_PATCHES_NUM / FIX_QUANTITY + 10)
sub_step = FIX_QUANTITY // BATCH_SIZE
EPOCHS = 10




PATH_PATCHES_VALIDATION = os.path.join(ROOT_DIR, '', 'patches/patches_validation')
PATH_PATCHES_TRAIN = os.path.join(ROOT_DIR, '', 'patches/patches_train_eval')
PATH_PATCHES_TEST = os.path.join(ROOT_DIR, '', 'patches/patches_test')
PATH_MASKS_VALIDATION = os.path.join(ROOT_DIR, '', 'masks/masks_validation')
PATH_MASKS_TRAIN = os.path.join(ROOT_DIR, '', 'masks/masks_train_eval')
PATH_MASKS_TEST = os.path.join(ROOT_DIR, '', 'masks/masks_test')

LENGTH_TRAIN = len(os.listdir(PATH_PATCHES_TRAIN))
LENGTH_VALIDATION = len(os.listdir(PATH_PATCHES_VALIDATION))
LENGTH_TEST = len(os.listdir(PATH_PATCHES_TEST))




EVALUATION_PATH = os.path.join(ROOT_DIR, '', 'evaluation/evaluation.txt')


print(PATH_MASKS_TRAIN)


