import cv2
import os

# Paths to the low-light and high-light image directories
low_light_dir = 'C:/Users/gaha/Documents/LOLdataset/our485/low'
high_light_dir = 'C:/Users/gaha/Documents/LOLdataset/our485/high'
combined_dir = 'C:/Users/gaha/Documents/LOLdataset/our485/combined'

# Create the combined image directory if it doesn't exist
os.makedirs(combined_dir, exist_ok=True)

# List of image filenames
low_light_images = sorted(os.listdir(low_light_dir))
high_light_images = sorted(os.listdir(high_light_dir))

# Ensure both directories have the same number of images
assert len(low_light_images) == len(high_light_images), "The number of images in both directories must be the same."


# Function to combine two images side by side
def combine_images(low_light_path, high_light_path, output_path, target_size=(256, 256)):
    # Read the images
    low_light_image = cv2.imread(low_light_path)
    high_light_image = cv2.imread(high_light_path)

    # Resize the images
    low_light_image = cv2.resize(low_light_image, target_size)
    high_light_image = cv2.resize(high_light_image, target_size)

    # Concatenate images horizontally
    combined_image = cv2.hconcat([low_light_image, high_light_image])

    # Save the combined image
    cv2.imwrite(output_path, combined_image)


# Combine each pair of images
for low_light_filename, high_light_filename in zip(low_light_images, high_light_images):
    low_light_path = os.path.join(low_light_dir, low_light_filename)
    high_light_path = os.path.join(high_light_dir, high_light_filename)
    output_filename = f"{low_light_filename}"
    output_path = os.path.join(combined_dir, output_filename)

    combine_images(low_light_path, high_light_path, output_path)
