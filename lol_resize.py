import cv2
import os

def process_images(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other image formats if needed
            # Read the image
            img = cv2.imread(os.path.join(input_dir, filename))

            # Check if the image has the correct dimensions
            if img.shape[0] == 400 and img.shape[1] == 1200:
                # Split the image into two parts
                left_part = img[:, :600]
                right_part = img[:, 600:]

                # Resize each part to 256x256
                left_part_resized = cv2.resize(left_part, (256, 256))
                right_part_resized = cv2.resize(right_part, (256, 256))

                # Concatenate the two parts horizontally
                concatenated_img = cv2.hconcat([left_part_resized, right_part_resized])

                # Save the resulting image in the output directory
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, concatenated_img)
            else:
                print(f"Skipping {filename}: Incorrect dimensions")

if __name__ == "__main__":
    input_directory = "train"  # Change this to the path of your input directory
    output_directory = "train_resized"  # Change this to the path of your output directory

    process_images(input_directory, output_directory)
