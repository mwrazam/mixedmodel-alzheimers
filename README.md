# Mixed-Input Neural Network with Naive Neural Architecture Search for Classification of Alzheimer's Disease in Brain MRI
In this work, we use a naive architecture search to find suitable models for a convolutional neural network and a multi-layer perceptron to classify patient metadata and brain MRI with corresponding Clinical Dementia Ratings. The differing model types are concatenated using a mixed-input model.

# Install
This project uses numerous packages with dependencies. At the time of development for this work, Tensorflow is only available for versions of Python up to 3.8. Python 3.9 is released but the current version of the Tensorflow backend for Keras is not compatible. 

It is HIGHLY recommended to use a virutal environment to setup all needed packages exactly as used during project development. This can be done on OSX/Linux by:

```
[PATH_TO_PYTHON_3.8.x]/python -m venv mm-ad
```

This will create a virtual environment with the right version of Python and Pip. Activate the environment:
```
source mm-ad/bin/activate
```

Next, you will likely have to update Pip to ensure it finds all of the correct packages:
```
pip install --upgrade pip
```

Now, you can use the requirements.txt file to install all of the correct required packages:
```
pip install -r requirements.txt
```

Once installation completes, you should have the required libraries to run the application. However, it is highly recommended that you setup the necessary toolkits to make use of an Nvidia GPU if you have one. If you don't have one, you can skip this section, but if you do, you'll need the following:

[CUDA] (https://developer.nvidia.com/cuda-toolkit) Enables GPU use for training and evaluation. Make sure to install the correct version for your GPU drivers!

[cuDNN] (https://developer.nvidia.com/cuDNN) Significantly speeds up convolution operations. You will require an Nvidia developer's login, but this is free and can be done pretty quickly. Make sure to install the version that matches your GPU drivers AND the toolkit you used!

Once everything is setup, the demo.py file contains a demonstration using some arbitrarily defined networks. Changing the "mode" in the __main__ function will change the execution mode. All experiments are contained in experiments.py, and similarly, the "mode" can be changed in the __main__ function to construct and execute experiments on the different types of models.
