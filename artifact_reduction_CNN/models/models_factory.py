from .models_architecture import *
from .model_interface import Model_Interface


# Build the model
class Models_Factory:
    def __init__(self):
        pass

    def make(self, cf):
        if cf.model_name == 'UNet_3D':
            model = UNet_3D(cf).make()
        elif cf.model_name == 'ResUNet_3D':
            model = ResUNet_3D(cf).make()
        elif cf.model_name == 'DnCNN_3D':
            model = DnCNN_3D(cf).make()
        elif cf.model_name == 'CDDN_3D':
            model = CDDN_3D(cf).make()
        else:
            raise ValueError('Unknown model name')

        return Model_Interface(model, cf)

