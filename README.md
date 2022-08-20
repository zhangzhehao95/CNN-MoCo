# CNN-MoCo

Motion-compensated (MoCo) reconstruction has shown great promise in proving the four-dimensional cone-beam computed tomography (4D-CBCT) image quality. Earlier MoCo methos uilize the motion model estimated from some prior planning CTs and all projections to perform 4D reconstruction for each phase. However, one problem is that the motion model derived from prior 4D-CT can be incorrect for CBCT since the breathing pattern and anatomy may change. So, it will be more desirable to build the motion model directly from CBCT data, like deriving the model from FDK-reconstructed 4D-CBCTs through some registration methods. However, the disadvantage is also apparent that there are lots of artifacts within these initial images. So the accuracy of motion model can not be guaranteed either. In this work, we propose to combine deep-learning based image processing and motion compensation by providing CNN-generated artifact-reduced initials images for MoCo to finally improve the 4D-CBCT image quality.

Here we provide the Python code for the artifact-reduction CNN training and inference and the registration parameters used for motion model estimation with the help of Elastix toolbox.

Authored by: Zhehao Zhang, Jiaming Liu, Deshan Yang, Ulugbek S. Kamilov, Geoffrey D. Hugo

# Run the code
## Prerequisites 
* tensorflow 2.0.0
* tensorflow_addons 0.6.0
* SimpleITK 2.0.2
* numpy 1.19.2

## Model and data
You can download the pre-trained model and one prepared SPARE challenge dataset in npy format from [Google Drive](https://drive.google.com/drive/folders/194KKJPdF-7xSAm5Z3YXO5LGKYzl6pFQw?usp=sharing). Once you have downloaded these files, You can also get more 4D-CBCT data from the [SPARE challenge website](https://image-x.sydney.edu.au/spare-challenge/).


## Usage
You will need a config file to 'direct' the code. By setting `train_model = True`, the training process is enabled. By setting `pred_model = True`, the code will compile the CNN model and load the pre-trained weights with the file name of `final_weights_file`. 

With the provided model weights 

```
python Selfcontained_CBCT/3Dpatch_code/main.py -c path_to_provided_config_file -d path_to_provided -e Selfcontained_CBCT/experiments/
```
