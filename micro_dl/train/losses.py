"""Custom losses"""
from keras import backend as K
import tensorflow as tf

import micro_dl.train.metrics as metrics
from micro_dl.utils.aux_utils import get_channel_axis


def mae_loss(y_true, y_pred, mean_loss=True):
    """Mean absolute error

    Keras losses by default calculate metrics along axis=-1, which works with
    image_format='channels_last'. The arrays do not seem to batch flattened,
    change axis if using 'channels_first
    """

    if not mean_loss:
        return K.abs(y_pred - y_true)

    channel_axis = get_channel_axis(K.image_data_format())
    return K.mean(K.abs(y_pred - y_true), axis=channel_axis)


def mse_loss(y_true, y_pred, mean_loss=True):
    """Mean squared loss"""

    if not mean_loss:
        return K.square(y_pred - y_true)

    channel_axis = get_channel_axis(K.image_data_format())
    return K.mean(K.square(y_pred - y_true), axis=channel_axis)


def kl_divergence_loss(y_true, y_pred):
    """KL divergence loss"""

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    if K.image_data_format() == 'channels_last':
        return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
    else:
        return K.sum(y_true * K.log(y_true / y_pred), axis=1)


def _split_ytrue_mask(y_true, n_channels):
    """Split the mask concatenated with y_true

    :param keras.tensor y_true: if channels_first, ytrue has shape [batch_size,
     n_channels, y, x]. mask is concatenated as the n_channels+1, shape:
     [[batch_size, n_channels+1, y, x].
    :param int n_channels: number of channels in y_true
    :return:
     keras.tensor ytrue_split - ytrue with the mask removed
     keras.tensor mask_image - bool mask
    """

    try:
        split_axis = get_channel_axis(K.image_data_format())
        y_true_split, mask_image = tf.split(y_true, [n_channels, 1],
                                            axis=split_axis)
        return y_true_split, mask_image
    except Exception as e:
        print('cannot separate mask and y_true' + str(e))


def masked_loss(loss_fn, n_channels):
    """Converts a loss function to mask weighted loss function

    Loss is multiplied by mask. Mask could be binary, discrete or float.
    Provides different weighting of loss according to the mask.
    https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
    https://github.com/keras-team/keras/issues/3270
    https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred

    nested functions -> closures
    A Closure is a function object that remembers values in enclosing
    scopes even if they are not present in memory. Read only access!!
    Histogram and logical operators are not differentiable, avoid them in loss
    modified_loss = tf.Print(modified_loss, [modified_loss],
                             message='modified_loss', summarize=16)
    :param Function loss_fn: a loss function that returns a loss image to be
     multiplied by mask
    :param int n_channels: number of channels in y_true. The mask is added as
     the last channel in y_true
    :return function masked_loss_fn
    """

    def masked_loss_fn(y_true, y_pred):
        y_true, mask_image = _split_ytrue_mask(y_true, n_channels)
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        loss = loss_fn(y_true, y_pred, mean_loss=False)
        mask = K.batch_flatten(mask_image)
        modified_loss = K.mean(loss * mask, axis=1)
        return modified_loss
    return masked_loss_fn


def dice_coef_loss(y_true, y_pred):
    """
    The Dice loss function is defined by 1 - DSC
    since the DSC is in the range [0,1] where 1 is perfect overlap
    and we're looking to minimize the loss.

    :param y_true: true values
    :param y_pred: predicted values
    :return: Dice loss
    """
    return 1. - metrics.dice_coef(y_true, y_pred)
