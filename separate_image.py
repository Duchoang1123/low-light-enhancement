import os
import cv2

# Define the paths
source_folder = 'val'
input_folder = 'Input_Image'
ground_truth_folder = 'Ground_Truth'

# Create directories if they don't exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(ground_truth_folder, exist_ok=True)

# Function to process each image
def process_image(file_path, input_folder, ground_truth_folder):
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error reading {file_path}")
        return
    
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Calculate the midpoint
    midpoint = width // 2
    
    # Split the image into left and right halves
    left_image = image[:, :midpoint]
    right_image = image[:, midpoint:]
    
    # Get the filename
    filename = os.path.basename(file_path)
    
    # Save the images to their respective folders
    cv2.imwrite(os.path.join(input_folder, filename), left_image)
    cv2.imwrite(os.path.join(ground_truth_folder, filename), right_image)

# Process all images in the source folder
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)
    
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        process_image(file_path, input_folder, ground_truth_folder)
    else:
        print(f"Skipping non-image file: {filename}")

print("Processing complete.")
