# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def lrFlip(x, y):
    # 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    x = tf.image.flip_left_right(x)
    y = tf.image.flip_left_right(y)
    return x, y


def udFlip(x, y):
    x = tf.image.flip_up_down(x)
    y = tf.image.flip_up_down(y)
    return x, y


def zoom(x, y, zoom_range=(1.0, 1.0), target_size=None):
    Minval, Maxval = zoom_range
    rate_a = tf.random.uniform(shape=[], minval=Minval, maxval=Maxval, dtype=tf.float32)
    rate_b = tf.random.uniform(shape=[], minval=Minval, maxval=Maxval, dtype=tf.float32)

    if len(target_size) == 4:
        target_height = target_size[1]
        target_width = target_size[2]
    elif len(target_size) == 3:
        target_height = target_size[0]
        target_width = target_size[1]

    # Resizes an image to a target size by keeping the aspect ratio the same without distortion (with padding).
    x = tf.image.resize_with_pad(x, tf.cast(target_height*rate_a, dtype=tf.int32), tf.cast(target_width*rate_b, dtype=tf.int32))
    y = tf.image.resize_with_pad(y, tf.cast(target_height*rate_a, dtype=tf.int32), tf.cast(target_width*rate_b, dtype=tf.int32))
    #  Centrally cropping the image or padding it evenly with zeros
    x = tf.image.resize_with_crop_or_pad(x, target_height, target_width)
    y = tf.image.resize_with_crop_or_pad(y, target_height, target_width)
    x = tf.clip_by_value(x, 0, 1)
    y = tf.clip_by_value(y, 0, 1)

    return x, y


def rotate(x, y, ro_range=0.0):
    pi = tf.constant(np.pi)
    ro_range = ro_range/180*pi

    angle = tf.random.uniform(shape=[], minval=-ro_range, maxval=ro_range, dtype=tf.float32)
    tfa.image.transform_ops.rotate(x, angle)
    tfa.image.transform_ops.rotate(y, angle)
    return x, y


def Gaussian_Noise(x, y, noise_mean=0.0, noise_std=1.0):
    noise = tf.random.normal(shape=tf.shape(x), mean=noise_mean, stddev=noise_std, dtype=tf.float32)
    x = tf.add(x, noise)
    y = tf.add(y, noise)
    return x, y


def shift(x, y, dx_range=0, dy_range=0):
    dx = tf.random.uniform(shape=[], minval=-dx_range, maxval=dx_range, dtype=tf.float32)
    dy = tf.random.uniform(shape=[], minval=-dy_range, maxval=dy_range, dtype=tf.float32)

    x = tfa.image.translate(x, [dx, dy], interpolation="BILINEAR")
    y = tfa.image.translate(y, [dx, dy], interpolation="BILINEAR")

    return x, y


def no_action(x, y):
    return x, y
