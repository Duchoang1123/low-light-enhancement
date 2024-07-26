# create three folders: Input_Image, Ground_Truth, and Predicted_Image

import os

# Define the paths
source_folder = 'val'
input_folder = 'Input_Image'
ground_truth_folder = 'Ground_Truth'
predicted_folder = 'Predicted_Image'

# Create directories if they don't exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(ground_truth_folder, exist_ok=True)
os.makedirs(predicted_folder, exist_ok=True)

