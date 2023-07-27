# Fish-Sense 
**Automated Biomass Estimation and Early Disease Detection in Aquaculture using computer vision**
# Introduction
We present a three-tiered process for automated  biomass estimation and disease detection as detailed in the Figure. The initial step involves generating a segmentation mask of the fish using Mask-RCNN, which aids in calculating crucial parameters such as length and height. Simultaneously, the fish species is identified leveraging a Convolutional Neural Network (CNN). In the next step, these parameters are incorporated to estimate the biomass. In addition we have trained a CNN module to classify fish into healthy and unhealthy categories, and subsequently identifying symptoms and locations of bacterial infections if a fish is classified as unhealthy. 

![Alt ](/Fig/Methodology_diagram.png)

# Abstract

The global demand for seafood is growing rapidly, making sustainable aquaculture an essential solution to address overfishing and meet food security challenges. Disease outbreaks and inaccurate biomass estimation pose significant challenges to the aquaculture industry.
In this paper, we propose a deep learning-based pipeline, developed in collaboration with fish farms, to automate fish disease detection and fish biomass estimation in aquaculture reservoirs.The automated framework is comprised of two modules. The first module focuses on fish biomass estimation, utilizing deep learning algorithms to segment fish, classify them into five species, and estimate their biomass. The second module aims at detecting disease symptoms, employing a deep learning algorithm to classify fish into healthy and unhealthy categories, and subsequently identifying symptoms and locations of bacterial infections if a fish is classified as unhealthy. To overcome the limited availability of data in this domain, we have created four novel real-world datasets for fish segmentation, healthy/unhealthy classification, fish species classification, and fish part segmentation, comprising 830, 2157, 5469, and 625 images, respectively. The developed biomass estimation algorithms demonstrate reliable accuracies, achieving 85 \% for perch, 86.5 \% for char, 93.3 \% for pikeperch, 96 \% for tilapia, and an impressive 99.6 \% for trout. Additionally, the classification of healthy and unhealthy fish achieves an accuracy of 94.4 \%. These algorithms provide a foundation for the development of industrial software solutions to improve fish health monitoring in aquaculture farms. By integrating our pipeline into software solutions, we bridge the gap between research and real-world applications, promoting sustainable aquaculture practices aligned with the United Nations' SDGs.

# Trained Models and Datasets

To run this project trained models can be obtained on permission from the authors by filling the form below:

https://docs.google.com/forms/d/e/1FAIpQLScHzbEzj97v6YZn3EdU8Pt4aMXj5cGPe4qJ05mQrM6df54tJg/viewform?usp=sf_link

Dataset used in this project can also be requested using same form. 

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
Once the virtual env is created install pytorch using command (For windows and CPU only)  
```bash 
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
Otherwise for GPU enabled environment and for linux OS, various installation commands can be found here https://pytorch.org/get-started/locally/ <br />
Install OpenCV using command. 

```bash
 pip install opencv-python 
```
Note:Pytorch and OpenCV are the prerequisite to install detectron2

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

Copy and paste this link in any of the supported browsers (Firefox, Chrome, and Microsoft Edge) by simply copying and pasting it into the address bar.

Upload an image and press Send button 
![Alt ](Fig\Demo_prototype.png)

The output will be processed image and resulting table 
![Alt ](Fig\Demo_prototype2.png)


# Authors 
Kanwal Aftab, Linda Tschirren, Boris Pasini, Peter Zeller, Bostan Khan and Muhammad Moazam Fraz
