# CNN-MoCo

Motion-compensated (MoCo) reconstruction has shown great promise in proving the four-dimensional cone-beam computed tomography (4D-CBCT) image quality. Earlier MoCo methos uilize the motion model estimated from some prior planning CTs and all projections to perform 4D reconstruction for each phase. However, one problem is that the motion model derived from prior 4D-CT can be incorrect for CBCT since the breathing pattern and anatomy may change. So, it will be more desirable to build the motion model directly from CBCT data, like deriving the model from FDK-reconstructed 4D-CBCTs through some registration methods. However, the disadvantage is also apparent that there are lots of artifacts within these initial images. So the accuracy of motion model can not be guaranteed either. In this work, we propose to combine deep-learning based image processing and motion compensation by providing CNN-generated artifact-reduced initials images for MoCo to finally improve the 4D-CBCT image quality.

Here we provide the Python code for the artifact-reduction CNN training and inference as well as the registration parameters used for motion model estimation with the help of Elastix toolbox.

Authored by: Zhehao Zhang, Jiaming Liu, Deshan Yang, Ulugbek S. Kamilov, Geoffrey D. Hugo

# Run the code
## Prerequisites 
* tensorflow 2.0.0
* tensorflow_addons 0.6.0
* SimpleITK 2.0.2
* numpy 1.19.2

## Model and data
You can download the pre-trained model and one processed SPARE challenge dataset in .npy format from [Google Drive](https://drive.google.com/drive/folders/194KKJPdF-7xSAm5Z3YXO5LGKYzl6pFQw?usp=sharing). Once you have downloaded these two folders (data & experiments), unzip and put them into the main folder (upper-level of the CNN code folder 'artifact_reduction_cnn').

You can also get more 4D-CBCT data from the [SPARE challenge website](https://image-x.sydney.edu.au/spare-challenge/).


## Usage
You will need a config file to direct the code. An example config has been provided (artifact_reduction_cnn/demo_config.py).

Setting `train_model = True`, the training process is enabled. By setting `pred_model = True`, the code will compile the CNN model and load the pre-trained weights file. 

You can run the demo to reduce the streaking artifacts from the SPARE images using the provided model: 

```
python artifact_reduction_cnn/main.py -c artifact_reduction_cnn/demo_config.py -d data/ -e experiments/
```

To train your own model, you can easily include your own data in the 'data' folder and modify the config file by setting `train_model = True` and changing `train_dir` correspondingly. 

# Motion model estimation
To reproduce our work, the outputs of artifact-reduction CNN (CNN enhanced images) are only the intermediate results. To get the final CNN+MoCo results, you will also need to estimate the motion model using the CNN enhanced images and utilize such motion model to performed a motion-compensated reconstruction.

We estimated the motion model using a groupwise registation method. That's to say, you will need to construct the moving and fixed images as 4D data, where the moving images is constructed by concatenating all ten phases together and fixed image is constructed by duplicating one specific phase ten times. We are using elastix v5.0.1 to perform the registation and the parameter file is provided.

# Motion-compensated reconstruction
After getting the motion model for one specific phase, we performed the MoCo reconstruction for that phase using a voxel-driven FDK algorithm which back-projected the projection data into the volume space along warped trajectories according to the phase-correlated deformation vector fields. Reconstruction Toolkit v2.1.0 were utilized for this final step.

# Contact
The code is provided to support reproducible research. If there is any unknown error or you need further help, please feel free to contact me at zhehao.zhang@wustl.edu.
