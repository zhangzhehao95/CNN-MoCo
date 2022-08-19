import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from .LR_schedule import Scheduler
import datetime

# Create callbacks
class Callbacks_Factory():
    def __init__(self):
        pass

    def make(self, cf):
        cb = []

        # Early stopping
        if cf.earlyStopping_enabled:
            print('Early stopping')
            cb += [EarlyStopping(monitor=cf.earlyStopping_monitor,
                                 mode=cf.earlyStopping_mode,
                                 patience=cf.earlyStopping_patience,
                                 verbose=cf.earlyStopping_verbose)]

        # Define model saving callbacks
        if cf.checkpoint_enabled:
            print('Model Checkpoint')
            cb += [ModelCheckpoint(filepath=os.path.join(cf.weights_save_path, "weights-{epoch:02d}.hdf5"),
                                   verbose=cf.checkpoint_verbose,
                                   monitor=cf.checkpoint_monitor,
                                   mode=cf.checkpoint_mode,
                                   save_freq=cf.checkpoint_save_freq_epoch,
                                   save_best_only=cf.checkpoint_save_best_only,
                                   save_weights_only=cf.checkpoint_save_weights_only)]

        # Learning rate scheduler
        if cf.LRScheduler_enabled:
            print('Learning rate scheduler by epoch')
            scheduler = Scheduler(cf.LRScheduler_type, cf.LRScheduler_power, cf.n_epochs, cf.learning_rate_base)
            # Scheduler function takes epoch(integer, indexed from 0) and current learning rate (float) as input
            cb += [LearningRateScheduler(scheduler.scheduler_function)]

        # Learning rate scheduler
        if cf.tensorBoard_enabled:
            print('TensorBoard for logging')
            log_dir = os.path.join(cf.savepath, 'tensorboard_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            cb += [tensorboard_callback]

        # Output the list of callbacks
        return cb
