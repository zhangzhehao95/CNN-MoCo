#!/usr/bin/env python
import os
import sys
import random
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from configuration.configuration import Configuration
from utils.logger import Logger
from dataset.dataset_generator import Dataset_Generator
from models.models_factory import Models_Factory
from callbacks.callbacks_factory import Callbacks_Factory

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


def main(json_path='configure_default.py'):
    parser = ArgumentParser(description='DnCNN training')
    parser.add_argument('-c', '--config_file', type=str, default=json_path, help='Path to config file', required=True)
    parser.add_argument('-d', '--data_path', type=str, help='Path to dataset', required=True)
    parser.add_argument('-e', '--experiment_path', type=str, help='Path to experiments (output)', required=True)
    args = parser.parse_args()

    # Load configuration files
    dataset_path = args.data_path
    experiments_path = args.experiment_path
    config_file = args.config_file
    configuration = Configuration(config_file, dataset_path, experiments_path)
    cf = configuration.load()

    if cf.train_model:
        # Enable log file
        sys.stdout = Logger(cf.log_file)

    # seed
    seed = cf.manual_seed
    if seed is None:
        seed = random.randint(1, 10000)
    print('Utilized random seed :' + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create the dataset generators
    print('\n > Generating dataset...')
    train_dataset, val_dataset, test_dataset, pred_dataset = Dataset_Generator().make(cf)

    # Build model
    print('\n > Building model...')
    model = Models_Factory().make(cf)
    
    # Create the callbacks
    print('\n > Creating callbacks...')
    cb = Callbacks_Factory().make(cf)

    # Training / testing
    try:
        if cf.train_model:
            # Train the model
            model.train(train_dataset, val_dataset, cb)

        if cf.test_model:
            model.test(test_dataset)

        if cf.pred_model:
            model.predict(pred_dataset)

    except KeyboardInterrupt:
        # User early stopping
        print('Stopped by user.')

    # Finish
    print(' ---> Finished experiment: ' + cf.experiment_name + ' <---')


if __name__ == '__main__':
    main()
