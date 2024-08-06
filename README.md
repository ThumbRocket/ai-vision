# Summary
This repository is vison model reference to make jupyter NoteBook

# Classfication
### CNN_Classification.ipynb
In this notebook, we will delve into advanced image classification techniques using PyTorch. We will cover four essential chapters, each focusing on a critical aspect of building, training, and utilizing state-of-the-art deep learning models for image classification tasks.
### Vison_Transfomer.ipynb
This notebook provides the simple walkthrough of the Vision Transformer. We hope you will be able to understand how it works by looking at the actual data flow during inference.
### Swin_Transformer.ipynb
Swin Transformer is a hirarchical vision transformer that was published in 2021 and selected as the best paper at ICCV 2021. In this post, we'll go over the fundamental concepts of Swin Transformer without the full explanation of implementation details (Although some provided).

# Segmentation
### Segmentation_FCN.ipynb
In this Notebook, we will perform Image Segmentation using Fully Convolutional Network on PASCAL VOC 2011 dataset which contains 20 object categories. We use the Semantic Boundaries Dataset (SBD) as it contains more segmentation labels than the original dataset.
### Segmentation_segmenter.ipynb
we will perform semantic segmentation on PASCAL VOC 2011 dataset which contains 20 object categories. We use the Semantic Boundaries Dataset (SBD) as it contains more segmentation labels than the original dataset

# Detection
### Faster-R-CNN.ipynb
try object detection using Faster R-CNN, a two-stage detector that uses medical mask detection.
### YOLOv5_Tutorial.ipynb
In this Notebook, try to run YOLOv5 Tutorial code (https://github.com/ultralytics/yolov5)
### DETR_Demo.ipynb
In this notebook, we show a demo of DETR (Detection Transformer), with slight differences with the baseline model in the paper. We show how to define the model, load pretrained weights and visualize bounding box and class predictions.
### DETR_hands_on.ipynb
In this notebook, we show-case how to use the pre-trained DETR models that we provide to make predictions and visualize the attentions of the model to gain insights on the way it sees the images.
### inference_for_YOLOv5.ipynb
we inference object using YOLOv5 pretrained model. and we compare slice inference with a YOLOv5 Model.

# 3D_Reconstruction
### homography.ipynb
In this notebook, explain Homography estimation using DLT 
### panorama.ipynb
This notebooks is Using DLT to compute homography from correspondences, and execute RANSAC(Random Sample Consensus) to find the homography which has largest correspondences (inliers) and genrate Panorama Image
### camera_model.ipynb
Thos notebook is to understand camera model, and make to create a renderer
### Triangulation.ipynb
In this notebook, we will triangulate facial landmarks of multiview face images, using landmarks acquired by Face-Alignment package.
### calibration_sfm_nerf_shared.ipynb
In here, implement the camera calibration process using Zhang's method.