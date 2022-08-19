from importlib.machinery import SourceFileLoader
import os
from os.path import join, exists
import shutil


class Configuration:
    def __init__(self, config_file,
                 dataset_path,
                 experiments_path):

        self.config_file = config_file
        self.dataset_path = dataset_path
        self.experiments_path = experiments_path

    def load(self):
        # Load configuration file
        print('Configuration file path: ' + self.config_file)
        cf = SourceFileLoader('config', self.config_file).load_module()

        # Save extra parameter
        cf.config_file = self.config_file
        cf.dataset_path = self.dataset_path
        cf.experiments_path = self.experiments_path

        # Create output folder
        cf.savepath = join(cf.experiments_path, cf.dataset_name, cf.experiment_name)
        if not exists(cf.savepath):
            os.makedirs(cf.savepath)

        # Define a log file and save print info into the log file
        cf.log_file = join(cf.savepath, "logfile.log")

        # Create weights save folder
        cf.weights_save_path = join(cf.savepath, 'saved_model')
        if not exists(cf.weights_save_path):
            os.makedirs(cf.weights_save_path)
        cf.weights_file = join(cf.weights_save_path, cf.final_weights_file)

        # Network parameters
        # Whether use droupout or not, default not
        if not hasattr(cf, "dropout"):
            cf.dropout = False
        # Whether use L2 weight dacay with 0.001 or not, default False
        if not hasattr(cf, "weight_decay"):
            cf.weight_decay = False
        # ReLU as activation by default
        if not hasattr(cf, "activation"):
            cf.activation = 'ReLU'
        # No batch normalization by default
        if not hasattr(cf, "norm"):
            cf.norm = False
        # Bottleneck mode for res-unet by default
        if not hasattr(cf, "bottleneck"):
            cf.bottleneck = True
        # Kernel size 3 by default
        if not hasattr(cf, "kernel_size"):
            cf.kernel_size = 3
        # No shuffle for prediction
        if cf.shuffle_pred:
            cf.shuffle_pred = False

        if not hasattr(cf, "learn_residual"):
            cf.learn_residual = False

        if not hasattr(cf, "downsample"):
            cf.downsample = 'MaxPooling'

        if not hasattr(cf, "upsample"):
            cf.upsample = 'TransposeConv'

        if not hasattr(cf, "loss"):
            cf.loss = 'L2'

        # Define model shape (interface for network) and loaded dataset shape
        cf.model_shape = cf.patch_size + (cf.channel_size,)
        cf.input_shape = cf.data_size + (cf.channel_size,)

        if not hasattr(cf, "volume_slices"):
            cf.volume_slices = 96

        # Dataset paths
        cf.train_path_full = os.path.join(cf.dataset_path, cf.train_dir)
        cf.valid_path_full = os.path.join(cf.dataset_path, cf.valid_dir)
        cf.test_path_full = os.path.join(cf.dataset_path, cf.test_dir)
        cf.pred_path_full = os.path.join(cf.dataset_path, cf.pred_dir)

        # Copy config file
        shutil.copyfile(cf.config_file, join(cf.savepath, "config.py"))

        return cf
