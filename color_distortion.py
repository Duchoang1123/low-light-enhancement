import numpy as np
import colour
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os

count = 0

def calculate_color_distortion(image1, image2):
    global count  # Declare count as a global variable
    # rgb to lab
    image1_lab = cv2.cvtColor(image1, cv2.COLOR_RGB2LAB)
    image2_lab = cv2.cvtColor(image2, cv2.COLOR_RGB2LAB)
    # calculate color distortion
    color_distortion = colour.delta_E(image1_lab, image2_lab, method='CIE 2000')
    # visualize heatmap
    
    # normalize the color distortion
    normalized_color_distortion = color_distortion / np.max(color_distortion)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_color_distortion, cmap='viridis', annot=False, fmt=".2f")
    plt.title('Difference Matrix Heatmap')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.savefig(f'{count}_heatmap.png')
    count += 1
    # plt.show()

    return color_distortion

def read_and_process_images(ground_truth_folder, generated_folder):
    # Load images
    ground_truth_images = [cv2.imread(os.path.join(ground_truth_folder, filename)) for filename in os.listdir(ground_truth_folder)]
    generated_images = [cv2.imread(os.path.join(generated_folder, filename)) for filename in os.listdir(generated_folder)]
    # Resize images
    ground_truth_images = [cv2.resize(img, (256, 256)) for img in ground_truth_images]
    generated_images = [cv2.resize(img, (256, 256)) for img in generated_images]
    return ground_truth_images, generated_images

def output_color_distortion(ground_truth_folder, generated_folder):
    global average_color_distortion
    ground_truth_images, generated_images = read_and_process_images(ground_truth_folder, generated_folder)
    for i in range(len(ground_truth_images)):
        average = np.mean(calculate_color_distortion(ground_truth_images[i], generated_images[i]))
        average_color_distortion += average/len(ground_truth_images)
    print(f'Average color distortion: {average_color_distortion}')

average_color_distortion = 0 
output_color_distortion('pix2pix-new/Ground_Truth', 'pix2pix-new/Predicted_Image')

