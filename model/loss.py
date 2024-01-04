# -*- coding: utf-8 -*-

"""
===========================================================================================================

Created on Thu Aug 10 2023
Latest update: 08-10-23 

@author: AnhND
===========================================================================================================
"""



from tensorflow import keras
from keras import backend as K







def dice_coef(y_true, y_pred, smooth=1e-2):
    """DICE coefficient

    Computes the DICE coefficient, also known as F1-score or F-measure.

    :param y_true: Ground truth target values.
    :param y_pred: Predicted targets returned by a model.
    :param smooth: Smoothing factor.
    :return: DICE coefficient of the positive class in binary classification.
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """DICE loss function

    Computes the DICE loss function value.

    :param y_true: Ground truth target values.
    :param y_pred: Predicted targets returned by a model.
    :return: Negative value of DICE coefficient of the positive class in binary classification.
    """
    return 1 - dice_coef(y_true, y_pred)