![Imperial Logo](Images/logowide.png)

<p align="center">
  <b> MSc Applied Computational Science and Engineering </b>   
</p>

<p align="center">
  Independent Research Project 2018/19
</p>

**Name**: Hameed Khandahari    
**CID**: 01069638


*Summary*:  A GAN was developed for the generation of time-series sensory data using the MotionSense dataset. 

## Table of contents
* [Project Description](#ProjectDescription)
* [Repository Structure](#RepositoryStructure)

* [Getting Started](#GettingStarted)
    * [Prerequisites](#Prerequisites)
    * [Installation](#Installation)

* [License](#License)
* [Acknowledgments](#Acknowledgments)

## Project Description <a name="ProjectDescription"></a>
This project is completed to answer the following research question:
* Is it possible to develop a GAN capable of generating realistic sensory data

## Repository Structure <a name="RepositoryStructure"></a>
The repository contains:
* `` MotionSense\``. This contains  part of the MotionSense dataset collected by M. Malekzadeh. The repository containing the original files can be found [here](https://github.com/mmalekzadeh/motion-sense)

* `` utils.py ``. Contains helper functions to create directories for storing data and images, obtaining random batches of data and plotting.

* `` DataLoader.py ``. A Python class containing methods for loading data from the ``.csv`` files in the ``MotionSense\`` folder and restructuring them such that they can be used to train the models.

* ``DataHandler.py``. A Python class containing methods for feature scaling (normalisation and standardisation and corresponding reverse transformations). 

* ``classifier.py``. Contains a classifier model which is trained on the MotionSense data to predict the activity that was being performed whilst the data being tested was collected. This is entirely on Malekzadeh's [implementation](https://github.com/mmalekzadeh/motion-sense/blob/master/codes/gen_paper_codes/1_MotionSense_Trial.ipynb)

* ``gan.py``. A Python class containing for the generative adversarial network. It contains methods to build the discriminator, the generator and then to combine these to make the GAN. Furthermore, it includes functionality to use the generative model to produce a sample of data

* ``trainer.py``. A Python class containing methods for training instances of the GAN class. It also has functionality for saving the progress of the training process and information regarding the loss at each epoch




## Getting Started <a name="GettingStarted"></a>
This package is capable of producing time-series sensory data based on the training data supplied by the MotionSense dataset.
To generate data of your own, run ``main.py``. To select the type of particular kind of data you want you can change the ``act_labels`` and ``features``.

* The ``act_labels`` are the types of activities being performed. You can select from ``dws`` (going downstairs), ``ups`` (going upstairs), ``wlk`` (walking), ``jog`` (jogging), ``sit`` (sitting) and ``std`` (standing). Include these codes as strings in the ``act_labels`` list which is then passed into the appropriate ``DataLoader`` methods. It is recommended that you only select one activity at a time for improved performance.

* The ``features`` are the types of sensory data you wish to generate. This project has focused solely on accelerometer data, i.e. ``userAcceleration.x``, ``userAcceleration.y`` and ``userAcceleration.z`` so caution is advised when generating the other kinds of data. The other 9 ``features`` are:

* Three attitude (orientation) sensors: ``attitude.roll``, ``attitude.pitch``, ``attitude.yaw``. 
* Three gravity sensors: ``gravity.x``, 	``gravity.y``, 	``gravity.z``. 
* Three rotation rate sensors: ``rotationRate.x``, ``rotationRate.y``,	``rotationRate.z	``. 
Include these codes as strings in the ``features`` list which is then passed into the appropriate ``DataLoader`` methods.

You must also define an ``expt_name`` (a string), a directory by this name will be created and the data and images generated will be stored here. The generator models which will be saved in the ``expt_name/models`` subdirectory, can be used to generate data using a random latent noise vector as its input. An example of this is is given in ``generate.py`` 

### Prerequisites <a name="Prerequisites"></a>

* ``numpy`` - version 1.16.4
* ``scipy`` - version 1.3.1
* ``matplotlib`` - version 3.0.3
* ``pandas`` - version 0.24.2
* ``keras`` - version 2.2.5
* ``tensorflow`` - 1.14.0
* ``scipy`` - version 1.3.1


### Installation <a name="Installation"></a>
To install, download the repository



## License <a name="License"></a>
This project is licensed under the MIT License - see the LICENSE.md file for details


## Acknowledgments <a name="Acknowledgments"></a>
* Dr H. Haddadi, the supervisor for the project
* Mr M. Malekzadeh, for advice during the project and for the MotionSense Dataset
* Erik Linder-Norén, whose GitHub repositories provided some [inspiration](https://github.com/eriklindernoren)
* Staff and Students in the Earth Science and Engineering Department at Imperial College London


