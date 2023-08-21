# Fish-Sense 
**Automated Biomass Estimation and Early Disease Detection in Aquaculture using computer vision**
# Introduction
We present a three-tiered process for automated  biomass estimation and disease detection as detailed in the Figure. The initial step involves generating a segmentation mask of the fish using Mask-RCNN, which aids in calculating crucial parameters such as length and height. Simultaneously, the fish species is identified leveraging a Convolutional Neural Network (CNN). In the next step, these parameters are incorporated to estimate the biomass. In addition we have trained a CNN module to classify fish into healthy and unhealthy categories, and subsequently identifying symptoms and locations of bacterial infections if a fish is classified as unhealthy. 

![Alt ](/Fig/Methodology_diagram.png)

# Abstract

With the burgeoning global demand for seafood, potential solutions like aquaculture are increasingly significant, provided they address issues like pollution and food security challenges in a sustainable manner. However, significant obstacles such as disease outbreaks and inaccurate biomass estimation underscore the need for optimized solutions. This paper proposes a deep learning-based pipeline, developed in conjunction with fish farms, aiming to enhance disease detection and biomass estimation in aquaculture reservoirs. Our automated framework is two-pronged: one module for biomass estimation using deep learning algorithms to segment fish, classify species, and estimate biomass; and another for disease symptom detection that classifies fish health status and identifies bacterial infection symptoms and locations in unhealthy fish. To circumvent data scarcity in this field, we curated four novel real-world datasets for fish segmentation, health classification, species classification, and fish part segmentation. Our biomass estimation algorithms demonstrated substantial accuracy across five species, and the health classification. These algorithms form the basis for developing software solutions to bolster fish health monitoring in aquaculture farms. Our integrated pipeline facilitates the transition from research to real-world applications, potentially encouraging responsible aquaculture practices. Nevertheless, these advancements must be seen as part of a comprehensive strategy aimed at improving the aquaculture industry's sustainability and efficiency, in line with the United Nations' Sustainable Development Goals' evolving interpretations.

# Trained Models and Datasets

To run this project trained models can be obtained on permission from the authors by filling the form below:

https://docs.google.com/forms/d/e/1FAIpQLScHzbEzj97v6YZn3EdU8Pt4aMXj5cGPe4qJ05mQrM6df54tJg/viewform?usp=sf_link

Dataset used in this project can also be requested using same form.
There are four datasets available that can be utilized for training deep learning models in various tasks:

- Fish Segmentation
- Species Classification
- Healthy/Unhealthy Classification
- Fish Body Part Segmentation

The code required to train the models from scratch can be found in the supporting materials, specifically under the "colab training - file" folder. By using this code, you can obtain the weights and set up the models as per your needs.

Alternatively, if you prefer to use pre-trained models, you can directly request them from the authors. Once obtained, simply place the trained models in the designated "trained_models" folder for seamless integration into your project.

## Requirements
-   Python > 3.7
-   Pytorch
-   opencv-python 4.8.0.74
-   Django==3.2.8
-   djangorestframework==3.12.4
-   imutil==0.3.4
-   imutils==0.5.4
-   tensorflow==2.13.0
-   scipy==1.11.1

# Installation Instructions
Create a new conda environment uisng command 

```bash 
conda create --name your_env_name python
```
After running this command, you can activate the virtual environment using:
```bash 
conda activate your_env_name
```
Once the virtual env is created install pytorch using command (For windows and CPU only):  
```bash 
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
Otherwise for GPU enabled environment or for linux OS or both, various installation commands can be found here: https://pytorch.org/get-started/locally/ <br />

Install OpenCV using command. 

```bash
 pip install opencv-python 
```
Note: Pytorch and OpenCV are the prerequisite to install detectron2

Install the detectron2 repository on local PC git clone 


```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install the libraries and modules listed in  requirement.txt file  

```bash 
pip install -r requirements.txt
```


## Steps To run the project

In your anaconda environment and workspace folder type the below command to run the project 
```bash
python manage.py runserver
```
If the libraries are installed correctly:
Proceed with the  generated link. http://127.0.0.1:8000/ 

Copy and paste this link in the address bar of any of the supported browsers (Firefox, Chrome, and Microsoft Edge).

Upload an image and press Send button. We have provided sample images for users in the "detection_labels/images/" folder.


![Alt ](/Fig/Demo_prototype.jpg)


The output will be processed image and a resulting table like the one below would be provided. 


![Alt ](/Fig/Demo_prototype2.jpg)


# Authors 
Kanwal Aftab, Linda Tschirren, Boris Pasini, Peter Zeller, Bostan Khan and Muhammad Moazam Fraz
