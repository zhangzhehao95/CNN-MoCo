import os
import numpy as np
from utils.helper import *


class Model_Interface:
    def __init__(self, model, cf):
        self.model = model
        self.cf = cf

    # Train the model
    def train(self, train_dataset, val_dataset, cb):
        if self.cf.train_model:
            print('\n > Training the model...')
            if self.cf.valid_model:
                hist = self.model.fit(train_dataset,
                                      epochs=self.cf.n_epochs,
                                      callbacks=cb,
                                      validation_data=val_dataset)
            else:
                hist = self.model.fit(train_dataset,
                                      epochs=self.cf.n_epochs,
                                      callbacks=cb)
            self.model.save_weights(self.cf.weights_file)
            print('\n > Training finished.')
            return hist
        else:
            return None

    # Test the model
    # Test is only used for showing the statistical loss/ metrics, use "predict" to generate real images
    def test(self, test_dataset):
        if self.cf.test_model:
            print('\n > Testing the model...')
            # Load best trained model
            print("Pre-trained weights: " + self.cf.weights_file)
            self.model.load_weights(self.cf.weights_file)
            results = self.model.evaluate(test_dataset)
            print(results)
        else:
            return None

    # Predict the model
    def predict(self, pred_dataset):
        if self.cf.pred_model:
            print('\n > Predicting the model...')
            test_save_path = os.path.join(self.cf.savepath, 'test_results')
            if not os.path.exists(test_save_path):
                os.makedirs(test_save_path)

            # Load the pre-trained model
            print("Pre-trained weights: " + self.cf.weights_file)
            self.model.load_weights(self.cf.weights_file)

            # Numpy array of predictions.: (phase_num*patch_num, z, x, y, channel), x-axis need to be flipped
            phase_num = self.cf.phase_num_pred

            volume_size = self.cf.data_size
            volume_slcs = volume_size[0]
            volume_rows = volume_size[1]
            volume_cols = volume_size[2]

            # Patches always cover all slices
            patch_rows = self.cf.patch_size[1]
            patch_cols = self.cf.patch_size[2]

            overlap = self.cf.patch_overlap
            gap_rows = patch_rows - overlap
            gap_cols = patch_cols - overlap
            patch_num_rows = 1 + int(np.ceil((volume_rows - patch_rows) / gap_rows))
            patch_num_cols = 1 + int(np.ceil((volume_cols - patch_cols) / gap_cols))

            aug_volume_rows = patch_rows + (patch_num_rows - 1) * gap_rows
            aug_volume_cols = patch_cols + (patch_num_cols - 1) * gap_cols

            aug_volume_size = (volume_slcs, aug_volume_rows, aug_volume_cols)

            aug_input_array = np.zeros(aug_volume_size).astype(np.float32)
            aug_output_array = np.zeros(aug_volume_size).astype(np.float32)
            aug_weight_array = np.zeros(aug_volume_size).astype(np.float32)

            sub = 0     # index for prediction case
            p = 0       # index for phase number of one case

            i = 0
            j = 0
            for pred_data in pred_dataset:
                output = np.squeeze(self.model.predict(pred_data))
                pred_data = np.squeeze(pred_data)

                aug_input_array[:, i*gap_rows:i*gap_rows+patch_rows, j*gap_cols:j*gap_cols+patch_cols] += pred_data
                aug_output_array[:, i*gap_rows:i*gap_rows+patch_rows, j*gap_cols:j*gap_cols+patch_cols] += output
                aug_weight_array[:, i*gap_rows:i*gap_rows+patch_rows, j*gap_cols:j*gap_cols+patch_cols] += 1

                j += 1
                if j == patch_num_cols:
                    i += 1
                    j = 0

                if i == patch_num_rows:
                    i = 0

                    if self.cf.save_input_itk:
                        aug_input_array = aug_input_array / aug_weight_array
                        input_array = crop3D_luCorner(aug_input_array, volume_size)
                        input_data_3D = np.swapaxes(input_array, 0, 1)
                        input_data_3D = np.flip(input_data_3D, 0)
                        save_as_itk(input_data_3D,
                                    os.path.join(test_save_path, 'Input' + str(sub) + '_phase_' + str(p) + '.mha'),
                                    self.cf)
                        aug_input_array *= 0

                    aug_output_array = aug_output_array / aug_weight_array
                    output_array = crop3D_luCorner(aug_output_array, volume_size)
                    data_3D = np.swapaxes(output_array, 0, 1)  # Convert Numpy matrix with dimension (x, z, y) to ITK
                    data_3D = np.flip(data_3D, 0)
                    save_as_itk(data_3D,
                                os.path.join(test_save_path, 'Output' + str(sub) + '_phase_' + str(p) + '.mha'),
                                self.cf)
                    aug_output_array *= 0
                    aug_weight_array *= 0

                    p += 1
                    if p == phase_num:
                        p = 0
                        sub += 1

        else:
            return None
