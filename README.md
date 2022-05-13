# UltrasoundImageAnalyser
This repository contains a small python script capable of analyzing the distribution of pixel values from a defined area of the image.

## Installation:
First clone the repository and set up your python environment.
This is done with the following terminal commands:

```
git clone https://github.com/BeSuter/UltrasoundImageAnalyser.git
cd UltrasoundImageAnalyser

conda env create -f environment.yml
conda env list
conda activate img_analyzer
```

Add all the images you ant to analyze to the UltrasoundImageAnalyser folder. Done! Now you are ready to analyze your images :D

## Example: 
Assume that you added the following images to the UltrasoundImageAnalyser folder: img1.jpeg, img2.jpeg, img3.png

Run the image analyzer with the following terminal command in order to get the average pixel brightness of the three images:
```
python us_image_analyzer.py --image img1.jpeg img2.jpeg img3.png
```

In case all the histograms annoy you, add "--NOplot" to the command ;)

Keyboard commands while cropping the image:
```
r --> undo whatever you did
k --> confirm and visualize the cropped area of the image (you can still undo this step by pressing 'r')
c --> calculate statistics. This can not be undone. Will automatically prompt the next image or the final statistical analysis. 
```


