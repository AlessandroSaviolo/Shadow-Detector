# Shadow Detector

This work is the final project for the course _Computer Vision_ that I attended during my master degree at Department of Information Engineering (Padova, Italy). See `report.pdf` for the report containing the description of the presented algorithm and the analysis of the produced results.

The purpose of this project is to implement a shadow detector.

## 1. Assumptions

The shadow detector is developed under the following assumptions:

- One strong source of light (dataset images are all outdoor images where the only source of light is the sun)

- Focus on ground shadows

## 2. Dataset

The following datasets have been used to test the performances of the proposed method:

- [LawnBowling](https://drive.google.com/open?id=1ImEiJeWE5nSepkQyOHGX90o8bUpgpyfh)

- [Black](https://drive.google.com/open?id=146jKk2x2GreUJfNYRGtkaNTTYpvtcwuA)

- [Nature](https://drive.google.com/open?id=1wSUHjC3bgSeaRjh690Ter3PETxZbdNWS)

- [Random](https://drive.google.com/open?id=1QB9PeDCGmEZBwDh27nDcw6yM3FU-c7p0)

## 3. Methodology

The features considered by the shadow detector for the comparison of different segments of the image are intensity, color information and texture.

To better present the different steps that compose the proposed algorithm, the output produced by each step is illustrated. In particular, the following image is considered:

<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/1.jpg" width="250">
 </p>

### Steps:

0.  Preprocessing: filters are applied to the image in order to get better results in the segmentation step

<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/2.jpg">
 </p>

1.  Segmentation: 𝑘means is used to segment the image

<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/3.jpg">
 </p>

2.  Intensity thresholding: image is converted to HSV color space and the intensity channel is thresholded

<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/4.jpg">
 </p>

3.  Candidates computation: segments belonging to the white components of the thresholded image (at point 2) are selected as shadow candidates

<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/5.png">
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/6.png">
</p>

4.  Candidates merging: “similar” candidates are merged (the illustrated matrix is used to identify candidates that need to be merged)

<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/8.png">
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/9.png">
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/10.png">
 </p>

5.  Candidates comparison: candidates are compared with their non-candidate neighbors and selected as shadow based on chromaticity and texture

<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/12.png">
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/13.png">
 </p>
 
 After computing these 5 steps, the algorithm outputs the detected shadows:
 
<p align="center"> 
    <img src="https://github.com/AlessandroSaviolo/Shadow-Detector/blob/master/steps-output/14.jpg">
 </p> 
