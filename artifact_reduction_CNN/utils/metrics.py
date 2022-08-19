import tensorflow as tf


def structural_similarity(y_true, y_pred):
    # ssim: input: [..., height, width, channel]; output: [..., x]
    value = tf.math.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))
    return value


def peak_snr(y_true, y_pred):
    value = tf.math.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1))
    return value
