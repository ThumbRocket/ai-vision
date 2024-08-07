import cv2
import numpy as np
import os

current_file_directory = os.path.dirname(os.path.abspath(__file__))

img1 = cv2.imread(os.path.join(current_file_directory, 'sample_1.png')) # (256, 256, 3) value 5~254
img2 = cv2.imread(os.path.join(current_file_directory, 'sample_2.png')) # (256, 256, 3) value 5~254

# Write your code for calculating pixel-level mae
# v2 - parallelization
# normalize every pixel to be unit vector
img1_normalized_per_pixel = img1 / np.linalg.norm(img1, axis=2)[:, :, None]
img2_normalized_per_pixel = img2 / np.linalg.norm(img2, axis=2)[:, :, None]

# result of dot product of unit vectors == cos_sim_map
cos_sim_map = np.sum(img1_normalized_per_pixel * img2_normalized_per_pixel, axis=2) 

# convert cos similarity to angular error
ae_map = np.arccos(cos_sim_map) * 180 / np.pi

mae = np.mean(ae_map)
print(mae)