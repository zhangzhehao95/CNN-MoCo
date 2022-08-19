from .tf_dataset import tfDataset


# Load datasets
class Dataset_Generator():
    def __init__(self):
        pass

    def make(self, cf):
        ds_train = None
        ds_validation = None
        if cf.train_model:
            # Load training set
            print('\n > Reading training set...')
            ds_train = tfDataset(data_path=cf.train_path_full,
                                 input_shape=cf.input_shape,
                                 phase_num=cf.phase_num_train,
                                 patch_size=cf.patch_size,
                                 patch_overlap=cf.patch_overlap,
                                 data_suffix=cf.data_suffix,
                                 phase_as_batch=cf.phase_as_batch,
                                 batch_size=cf.batch_size_train,
                                 shuffle=cf.shuffle_train,
                                 aug=cf.train_aug,
                                 flag='train')

            print('\n > Reading validation set...')
            ds_validation = tfDataset(data_path=cf.valid_path_full,
                                      input_shape=cf.input_shape,
                                      phase_num=cf.phase_num_valid,
                                      patch_size=cf.patch_size,
                                      patch_overlap=cf.patch_overlap,
                                      data_suffix=cf.data_suffix,
                                      phase_as_batch=False,
                                      batch_size=cf.batch_size_valid,
                                      shuffle=cf.shuffle_valid,
                                      aug=False,
                                      flag='validation')

        ds_test = None
        if cf.test_model:
            # Load testing set
            print('\n > Reading testing set...')
            ds_test = tfDataset(data_path=cf.test_path_full,
                                input_shape=cf.input_shape,
                                phase_num=cf.phase_num_test,
                                patch_size=cf.patch_size,
                                patch_overlap=cf.patch_overlap,
                                data_suffix=cf.data_suffix,
                                phase_as_batch=False,
                                batch_size=cf.batch_size_test,
                                shuffle=cf.shuffle_test,
                                aug=False,
                                flag='test')

        ds_predict = None
        if cf.pred_model:
            # Load dataset to predict
            print('\n > Reading prediction set...')
            ds_predict = tfDataset(data_path=cf.pred_path_full,
                                   input_shape=cf.input_shape,
                                   phase_num=cf.phase_num_pred,
                                   patch_size=cf.patch_size,
                                   patch_overlap=cf.patch_overlap,
                                   data_suffix=cf.data_suffix,
                                   phase_as_batch=False,
                                   batch_size=cf.batch_size_pred,
                                   shuffle=False,
                                   aug=False,
                                   flag='pred')

        return ds_train, ds_validation, ds_test, ds_predict
