import tensorflow as tf
import os
import numpy as np
from .data_augmentation import lrFlip, udFlip, zoom, rotate, Gaussian_Noise, no_action


def load_npy(item):
    # Load 3D numpy array
    data = np.squeeze(np.load(item.numpy()))
    data = data[..., np.newaxis]
    tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    return tensor


def read_file_double(input, output, input_shape):
    input_data = tf.py_function(load_npy, [input], tf.float32)
    input_data.set_shape(input_shape)
    output_data = tf.py_function(load_npy, [output], tf.float32)
    output_data.set_shape(input_shape)
    return input_data, output_data


def read_file_single(input, input_shape):
    input_data = tf.py_function(load_npy, [input], tf.float32)
    input_data.set_shape(input_shape)
    return input_data


def make_tf_dataset(files_dataset, input_shape, is_pred=False):
    if is_pred:
        image_dataset = files_dataset.map(lambda input: read_file_single(input, input_shape))
    else:
        image_dataset = files_dataset.map(lambda input, output: read_file_double(input, output, input_shape))
    return image_dataset


def patching(image, patch_size, patch_overlap, phase_batch=True):
    if len(image.shape) < 5:
        image = tf.expand_dims(image, 0)
    # image.shape: [b,z,x,y,1]
    row = image.shape[2]
    col = image.shape[3]

    # Data format: [Batch, ..., Channel], always cover all slices
    gap_rows = patch_size[1] - patch_overlap
    gap_cols = patch_size[2] - patch_overlap
    ksizes = (1,) + patch_size + (1,)
    strides = (1, patch_size[0], gap_rows, gap_cols, 1)

    re_row = np.mod((row - patch_size[1]), gap_rows)
    re_col = np.mod((col - patch_size[2]), gap_cols)
    row_pad = 0 if re_row == 0 else gap_rows - re_row
    col_pad = 0 if re_col == 0 else gap_cols - re_col

    image = tf.pad(image, [[0, 0], [0, 0], [0, row_pad], [0, col_pad], [0, 0]], 'REFLECT')

    patches = tf.extract_volume_patches(image, ksizes, strides, 'VALID')

    # Patch with same location for all phases are stacked together
    if phase_batch:
        patches = tf.transpose(patches, perm=[1, 2, 3, 0, 4])

    image_tensor = tf.reshape(patches, [-1, patch_size[0], patch_size[1], patch_size[2]])
    image_tensor = image_tensor[..., tf.newaxis]
    return image_tensor


def tfDataset(data_path, input_shape, phase_num, patch_size, patch_overlap, data_suffix, phase_as_batch, batch_size,
              shuffle, aug, flag):
    # (z, x, y), 3D training by phase + slice
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    input_data_path = []
    if flag == 'pred':
        # Prediction (without known output)
        for sub_dir in os.listdir(data_path):
            subject_path = os.path.join(data_path, sub_dir)
            if os.path.isdir(subject_path):
                for p in range(phase_num):
                    input_file_name = 'p' + str(p) + data_suffix
                    input_file_list = os.path.join(subject_path, input_file_name)
                    input_data_path.append(input_file_list)

        ds_len = len(input_data_path)
        print('Number of ' + flag + ' pairs:' + str(ds_len))
        files_dataset = tf.data.Dataset.from_tensor_slices(input_data_path)
        images_ds = make_tf_dataset(files_dataset, input_shape, is_pred=True)
        images_ds = images_ds.flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(patching(x, patch_size, patch_overlap)))
        images_ds = images_ds.batch(batch_size)

    else:
        # Training, validation and testing
        output_data_path = []
        for sub_dir in os.listdir(data_path):
            subject_path = os.path.join(data_path, sub_dir)
            if os.path.isdir(subject_path):
                output_file_name = '3D' + data_suffix
                for p in range(phase_num):
                    input_file_name = 'PA_p' + str(p) + data_suffix
                    input_file_list = os.path.join(subject_path, input_file_name)
                    output_file_list = os.path.join(subject_path, output_file_name)

                    input_data_path.append(input_file_list)
                    output_data_path.append(output_file_list)

        ds_len = len(input_data_path)
        print('Number of ' + flag + ' pairs:' + str(ds_len))
        all_buf_size = ds_len * int(input_shape[1] // patch_size[1]) ** 2

        files_dataset = tf.data.Dataset.from_tensor_slices((input_data_path, output_data_path))
        images_ds = make_tf_dataset(files_dataset, input_shape)
        if phase_as_batch:
            images_ds = images_ds.batch(phase_num)
            images_ds = images_ds.flat_map(
                lambda x, y: tf.data.Dataset.from_tensor_slices((patching(x, patch_size, patch_overlap),
                                                                 patching(y, patch_size, patch_overlap))))

            images_ds = images_ds.batch(batch_size)

            if shuffle:
                buf_size = int(all_buf_size // batch_size // 2)
                images_ds = images_ds.shuffle(buffer_size=buf_size, reshuffle_each_iteration=True)

        else:
            images_ds = images_ds.flat_map(
                lambda x, y: tf.data.Dataset.from_tensor_slices((patching(x, patch_size, patch_overlap, False),
                                                                 patching(y, patch_size, patch_overlap, False))))

            if shuffle:
                buf_size = int(all_buf_size // 2)
                images_ds = images_ds.shuffle(buffer_size=buf_size, reshuffle_each_iteration=True)

            images_ds = images_ds.batch(batch_size)

        if aug:
            images_ds = images_ds.map(lambda x, y: tf.cond(tf.random.uniform([], 0, 1) > 0.8,
                                                           lambda: lrFlip(x, y),
                                                           lambda: no_action(x, y)))
            images_ds = images_ds.map(lambda x, y: tf.cond(tf.random.uniform([], 0, 1) > 0.8,
                                                           lambda: udFlip(x, y),
                                                           lambda: no_action(x, y)))
            images_ds = images_ds.map(lambda x, y: tf.cond(tf.random.uniform([], 0, 1) > 0.8,
                                                           lambda: zoom(x, y, zoom_range=[0.8, 1.2],
                                                                        target_size=input_shape),
                                                           lambda: no_action(x, y)))
            images_ds = images_ds.map(lambda x, y: tf.cond(tf.random.uniform([], 0, 1) > 0.8,
                                                           lambda: rotate(x, y, 20),
                                                           lambda: no_action(x, y)))
            images_ds = images_ds.map(lambda x, y: tf.cond(tf.random.uniform([], 0, 1) > 0.9,
                                                           lambda: Gaussian_Noise(x, y, 0, 1),
                                                           lambda: no_action(x, y)))

    print(images_ds)
    images_ds = images_ds.prefetch(buffer_size=AUTOTUNE)
    return images_ds
