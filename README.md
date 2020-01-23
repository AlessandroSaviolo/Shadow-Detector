# Shadow Detector

This work is the final project for the course _Computer Vision_ that I attended during my master degree at Department of Information Engineering (Padova, Italy). See `report.pdf` for the report containing the description of the presented algorithm and the analysis of the produced results.

The purpose of this project is to implement a shadow detector.

The shadow detector is developed under the following assumptions:

- One strong source of light (dataset images are all outdoor images where the only source of light is the sun)

- Focus on ground shadows

The features considered by the shadow detector for the comparison of different segments of the image are intensity, color information and texture.

The shadow detector method is divided in the following steps:

0. Preprocessing: filters are applied to the image in order to get better results in the segmentation step

1. Segmentation: 𝑘means is used to segment the image

2. Intensity thresholding: image is converted to HSV color space and the intensity channel is thresholded

3. Candidates computation: segments belonging to the white components of the thresholded image (at point 2) are selected as shadow candidates

4. Candidates merging: “similar” candidates are merged

5. Candidates comparison: candidates are compared with their non-candidate neighbors and selected as shadow based on chromaticity and texture

## 1. Dataset

- [SVHN Training set](http://ufldl.stanford.edu/housenumbers/train.tar.gz)

- [SVHN Test set](http://ufldl.stanford.edu/housenumbers/test.tar.gz)

## 2. Project Structure

- `load_data.py` : load the dataset and create the Pascal VOC annotations

- `train.py` : train a new model from scratch, or load the [model](https://drive.google.com/open?id=1-MqGpht6UnGzX3Ps_8-IIJ8hAz24pHoN) and keep training it

- `infer.py` : make predictions and output a JSON file containing the generated bounding box coordinates, labels and scores

- `RetinaNET.ipynb` : test inference speed of the model

## 3. Credits

The following GitHub Repositories have helped the development of this project:

- [Pavitrakumar78 Repository](https://github.com/pavitrakumar78/Street-View-House-Numbers-SVHN-Detection-and-Classification-using-CNN): the python file "construct datasets.py" has been used to parse the input file containing the dataset and to create the annotations

- [Fizyr Repository](https://github.com/fizyr/keras-retinanet): the entire repository has been imported and widely used to create and train the RetinaNET model

- [Penny4860 Repository](https://github.com/penny4860/retinanet-digit-detector): the pre-trained model has been taken from this repository
