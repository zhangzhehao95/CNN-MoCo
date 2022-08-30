# Experiment
dataset_name                 = 'SPARE_demo'     # Name of utilized dataset
experiment_name              = 'pre_trained'    # Experiment name under specific dataset
model_name                   = 'UNet_3D'


# Run instructions
train_model                  = False             # Train the model
valid_model                  = False             # Do validation during training
test_model                   = False             # Test the model
pred_model                   = True              # Predict using the model
manual_seed                  = 0                 # Manual random seed for code [|None]


# Dataset
train_dir                    = 'Train_npy_3D'                 # Directory of training dataset
valid_dir                    = 'Val_npy_3D'                   # Directory of validation dataset
test_dir                     = 'SPARE_test_npy_3D'            # Directory of test dataset
pred_dir                     = 'SPARE_test_npy_3D'            # Directory of predict dataset
data_suffix                  = '.npy'


# Data info
data_size                    = (96, 224, 224)         # Size of npy file
channel_size                 = 1


# Patch
patch_size                   = (96, 64, 64)       # All cover all slices
patch_overlap                = 6


# Phase
phase_num_train              = 10                 # Phase number of each subject for training data
phase_num_valid              = 10
phase_num_test               = 10
phase_num_pred               = 10


# ITK form to save
save_input_itk = True
spacing = (2, 2, 2)
origin = (-223, -95, -223)
direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


# Network parameters
loss                         = 'L2'                # 'L1', 'L2'
kernel_size                  = 3                   # Default 3
norm                         = 'Batch'		       # Default False
learn_residual               = True

# For DnCNN
filter_num                   = 64                  # Default 64
net_depth                    = 10                  # Default 10

# For Unet
filter_num_base              = 64
up_down_times                = 3
conv_times                   = 2
downsample                   = 'MaxPooling'
upsample                     = 'UpSample'
activation                   = 'ReLU'
weight_decay                 = False
dropout                      = False
bottleneck                   = True     # Res-unet

# For RDN
RDB_layer                    = 5                   # Number of layers in each RDB blocks
RDB_count                    = 2                   # Number of RDB blocks
alpha                        = 0.9

# For RED_CNN
REDCNN_filter_num            = 96


# Training parameters
optimizer                    = 'adam'                    # Optimizer ['sgd' | 'adam' | 'nadam']
learning_rate_base           = 1e-3                     # Base learning rate
n_epochs                     = 100                      # Number of epochs during training
final_weights_file           = 'final_weights.hdf5'     # File name for the final weights of model
train_aug                    = False                    # Using data augment  


# Batch sizes
phase_as_batch               = False            # Use phase number as batch size
batch_size_train             = 5              # Batch size during training
batch_size_valid             = 1              # Batch size during validation
batch_size_test              = 1              # Batch size during testing
batch_size_pred              = 1               # Batch size during prediction


#Shuffle
shuffle_train                = True            # Whether to shuffle the training dataset
shuffle_valid                = False           # Whether to shuffle the validation dataset
shuffle_test                 = False           # Whether to shuffle the testing dataset
shuffle_pred                 = False           # Whether to shuffle the prediction dataset


# Callback early stoping
earlyStopping_enabled        = False           # Enable the Callback
earlyStopping_monitor        = 'val_loss'      # Loss to monitor
earlyStopping_mode           = 'min'           # Mode ['max' | 'min' | 'auto']
earlyStopping_patience       = 50              # Max patience for the early stopping
earlyStopping_verbose        = 0               # Verbosity of the early stopping


# Callback model check point
checkpoint_enabled           = True            # Enable the Callback
checkpoint_monitor           = 'val_loss'      # Loss to monitor (Metric to monitor)
checkpoint_mode              = 'min'           # Mode ['max' | 'min'| 'auto']
checkpoint_save_best_only    = False           # Save best or last model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 1               # Verbosity of the checkpoint
checkpoint_save_freq_epoch   = 'epoch'         # Frequency of saving checkpoint, every epoch or every integral batches


# Callback learning rate scheduler
LRScheduler_enabled          = True                          # Enable the Callback
LRScheduler_type             = 'step_exp_decay'              # Type of scheduler ['exp_decay' | 'power_decay' | ...]
LRScheduler_power            = 0.9                           # Power for the power_decay, exp_decay modes


# Callback tensorboard
tensorBoard_enabled          = True            # Enable the Callback
