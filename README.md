# Summary
This repository is vison model reference to make jupyter NoteBook

# Classfication
### `CNN_Classification.ipynb`
In this notebook, we will delve into advanced image classification techniques using PyTorch. We will cover four essential chapters, each focusing on a critical aspect of building, training, and utilizing state-of-the-art deep learning models for image classification tasks.
### `Vison_Transfomer.ipynb`
This notebook provides the simple walkthrough of the Vision Transformer. We hope you will be able to understand how it works by looking at the actual data flow during inference.
### `Swin_Transformer.ipynb`
Swin Transformer is a hirarchical vision transformer that was published in 2021 and selected as the best paper at ICCV 2021. In this post, we'll go over the fundamental concepts of Swin Transformer without the full explanation of implementation details (Although some provided).

# Segmentation
### `Segmentation_FCN.ipynb`
In this Notebook, we will perform Image Segmentation using Fully Convolutional Network on PASCAL VOC 2011 dataset which contains 20 object categories. We use the Semantic Boundaries Dataset (SBD) as it contains more segmentation labels than the original dataset.
### `Segmentation_segmenter.ipynb`
we will perform semantic segmentation on PASCAL VOC 2011 dataset which contains 20 object categories. We use the Semantic Boundaries Dataset (SBD) as it contains more segmentation labels than the original dataset

# Detection
### `Faster-R-CNN.ipynb`
try object detection using Faster R-CNN, a two-stage detector that uses medical mask detection.
### `YOLOv5_Tutorial.ipynb`
In this Notebook, try to run YOLOv5 Tutorial code (https://github.com/ultralytics/yolov5)
### `DETR_Demo.ipynb`
In this notebook, we show a demo of DETR (Detection Transformer), with slight differences with the baseline model in the paper. We show how to define the model, load pretrained weights and visualize bounding box and class predictions.
### `DETR_hands_on.ipynb`
In this notebook, we show-case how to use the pre-trained DETR models that we provide to make predictions and visualize the attentions of the model to gain insights on the way it sees the images.
### `inference_for_YOLOv5.ipynb`
we inference object using YOLOv5 pretrained model. and we compare slice inference with a YOLOv5 Model.

# 3D_Reconstruction
### `homography.ipynb`
In this notebook, explain Homography estimation using DLT 
### `panorama.ipynb`
This notebooks is Using DLT to compute homography from correspondences, and execute RANSAC(Random Sample Consensus) to find the homography which has largest correspondences (inliers) and genrate Panorama Image
### `camera_model.ipynb`
Thos notebook is to understand camera model, and make to create a renderer
### `Triangulation.ipynb`
In this notebook, we will triangulate facial landmarks of multiview face images, using landmarks acquired by Face-Alignment package.
### `calibration_sfm_nerf_shared.ipynb`
In here, implement the camera calibration process using Zhang's method.
### `3D_Reconstruction.ipynb`
In here, implement Dense 3D Reconstruction and Volumetric Surface Reconstruction.
### `SMPL.ipynb`
In here, implement SMPL what is a statistical template model of naked human body.
### `PIFuHD.ipynb`
In here, implement PIFuHD model.

# Computational_Photography
### _wb : White Balance_
### `./wb/gray_world`
It is implemented grey world algorithm. moving grey world folder, you can run grey algorithm.
```bash
cd ./Computational_Phtography/wb/gray_world
python ./gray_world.py # script run
```

### `./wb/fc4`
It is implemented Pretrained Deep Model(fc4-https://github.com/yuanming-hu/fc4). moving fc4 folder, you can run fc4 model.
```bash
cd ./Computational_Phtography/wb/fc4
python ./test.py # script run
```

### `./wb/anglular_error`
It is implemented angular error. this code calculate angular error for the given rgb vector pair.
```bash
cd ./Computational_Phtography/wb/anglular_error/pixel_level_ae
python ./cal_ae_pixel.py # script run
```

### `./wb/LSMI_data_generation`
It is implemented LSMI data generation. LSMI data generation is A method of creating datasets by generating a large number of diverse instances.
```bash
cd ./Computational_Phtography/wb/LSMI_data_generation
python ./LSMI_data_gen_toy.py # script run
```

### `./wb/hdrnet`
It is implemented white balance using hdrnet.
```bash
cd ./Computational_Phtography/wb/hdrnet
python ./python test.py --dataset ./data_sample/ --ckpt ch/pretrained_model.pth --bit-depth 10 --net-input-size 128 --net-output-size 256 # script run
```



### `./wb/unet`
It is implemented white balance using UNet.
```bash
cd ./Computational_Phtography/wb/unet
python main.py --mode test --data_root data_sample/ --model_root checkpoint/ --checkpoint pretrained_unet/model.pt --output_type illumination --save_result 'yes' # script run
```

### _sr : Super Resolution_
### `./sr/interpolation`
In this code, implemented free SR on Classical Training
```bash
cd ./Computational_Phtography/sr/interpolation
python ./interpolation.py # script run
```

### `./sr/deep_sr`
In this code, implemented Enhanced Deep Residual Network
```bash
cd ./Computational_Phtography/sr/deep_sr
pip install -r requirements.txt
python test_edsr.py --config configs/test_edsr-baseline-multi.yaml # script run
```

### `./sr/deep_sr`
In this code, implemented Local Implicit Image Function
```bash
cd ./Computational_Phtography/sr/deep_sr
pip install -r requirements.txt
python test_liif.py --config configs/test_edsr-baseline-liif.yaml # script run
```
# Motion tracking
### `Optical_flow_LK.ipynb`
In this notebook, The Lucas-Kanade Optical Flow for Optical Flow estimation has been implemented.
### `Optical_flow_RAFT.ipynb`
In this notebook, RAFT(Recurruent All-pairs Field Transfomers for Optical Flow Estimation) using Deep Learning has been implemented.
### `Dense_Optical_Tracking.ipynb`
In this notebook, Dense Optical Tracking calculates the motion of every pixel in an image sequence, providing a comprehensive motion field. It is useful in applications like video stabilization and motion detection, offering detailed motion information across the entire image.
### `Kalman_Filter.ipynb`
In this notebook, implement Kalman Filter. The Kalman Filter is an algorithm that estimates the state of a dynamic system by combining predictions and noisy measurements to produce an optimal estimate.
### `3D_Object_Tracking.ipynb`
In this notebook, implement 3D Object Tracking notebook