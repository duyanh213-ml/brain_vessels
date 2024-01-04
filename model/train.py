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
from model_architect import *

import tensorflow_model_optimization as tfmot


import tempfile
import numpy as np
import tifffile as tiff


# Utilizing function: This function is constructed to avoid run out of memory

def mini_batch():
    index = np.random.choice(
        range(TOTAL_PATCHES_NUM), size=FIX_QUANTITY, replace=False)

    np.random.shuffle(index)

    imgs_input = []
    imgs_output = []

    count = 1

    for i in index:
        patch_path = f"{ROOT_DIR}/patches/patches_train_train/"
        mask_path = f"{ROOT_DIR}/masks/masks_train_train/"

        imgs_input.append(tiff.imread(f"{patch_path}slice_patch_{i}.tiff"))
        imgs_output.append(tiff.imread(f"{mask_path}mask_patch_{i}.tiff"))

        count += 1

    print(
        f"Successfully load random with {FIX_QUANTITY} patch and mask index to array")
    print(index)

    imgs_input = np.array(
        imgs_input, dtype=np.float32).reshape(-1, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS)
    imgs_output = np.array(
        imgs_output, dtype=np.float32).reshape(-1, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS)

    return imgs_input, imgs_output


# Define the model from it's architecture and compile it


if __name__ == "__main__":

    model = unet_2D(INPUT_SHAPE, FILTERS_LIST, KERNEL_SIZE, DOWN_STRIDES, UP_STRIDES,
                    PADDING, ACTIVATION, FINAL_ACTIVATION, KERNEL_INITIALIZER, RATE)

    model.compile(optimizer=OPTIMIZER, loss=LOSS)

    for epoch in range(EPOCHS):

        print(f"EPOCH {epoch + 1}")

        for step in range(STEPS_PER_EPOCH):
            print(f"Step {step + 1}/{STEPS_PER_EPOCH}:")
            imgs_input, imgs_output = mini_batch()

            history = model.fit(imgs_input, imgs_output,
                                epochs=1, batch_size=BATCH_SIZE)

        # One more step to save loss
        imgs_input, imgs_output = mini_batch()

        print("BONUS Step:")
        history = model.fit(imgs_input, imgs_output,
                            epochs=1, batch_size=BATCH_SIZE)

        with open(LOSSES_SAVE_PATH, "a") as f:
            f.write(f"\n{history.history['loss'][-1]}")

        #  After end 1 epoch: save model
        model.save(SAVE_MODEL_PATH)
        print(f"Successfully saving to {SAVE_MODEL_PATH}")

    # This code below use to prunning our model to become smaller in the size

    sub_step = FIX_QUANTITY // BATCH_SIZE

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.00,
                                                                 final_sparsity=0.40,
                                                                 begin_step=0,
                                                                 end_step=sub_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer=OPTIMIZER, loss=dice_coef_loss)

    model_for_pruning.summary()

    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    imgs_input, imgs_output = mini_batch()
    model_for_pruning.fit(imgs_input, imgs_output,
                          batch_size=BATCH_SIZE, epochs=1,
                          callbacks=callbacks)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_for_export.save(SAVE_PRUNNING_MODEL_PATH)
    