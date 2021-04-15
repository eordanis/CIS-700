![SU College Logo](Images/SU_LOGO.jpg)

<p align="center">
  <b> MS Syracuse University College of Engineering and Computer Science </b>   
</p>

<p align="center">
  Machine Learning and Security Masters Research Project Milestone 04/07/21-06/16/21
</p>

**Name**: Stephanie Eordanidis
		  Ravjot Sachdev
		  Jackson Taber


*Summary*:  A previously developed GAN was developed and repurposed for the use of adversarial text generation. 

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
* Is it possible to develop a GAN capable of generating realistic/synthetic text data


## Repository Structure <a name="RepositoryStructure"></a>
The repository contains:
* `` Data\``. This contains  part of the TextData dataset obtained via kaggle source -> [url here]

* `` utils.py ``. Contains helper functions to create directories for storing data and images, obtaining random batches of data and plotting.

* `` DataLoader.py ``. A Python class containing methods for loading data from the ``.csv`` files in the ``TextData\`` folder and restructuring them such that they can be used to train the models.

* ``DataHandler.py``. A Python class containing methods for feature scaling (normalisation and standardisation and corresponding reverse transformations). 

* ``classifier.py``. Contains a classifier model which is trained on the TextData data to predict the activity that was being performed whilst the data being tested was collected.

* ``gan.py``. A Python class containing for the generative adversarial network. It contains methods to build the discriminator, the generator and then to combine these to make the GAN. Furthermore, it includes functionality to use the generative model to produce a sample of data

* ``trainer.py``. A Python class containing methods for training instances of the GAN class. It also has functionality for saving the progress of the training process and information regarding the loss at each epoch


## Getting Started <a name="GettingStarted"></a>
*TBD*

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
*None*


## Acknowledgments <a name="Acknowledgments"></a>
This work's foundation was provided by the github project located at url https://github.com/msc-acse/acse-9-independent-research-project-hk-97


