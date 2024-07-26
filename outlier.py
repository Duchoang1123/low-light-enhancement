import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Function to load and preprocess stitched images
def load_and_preprocess_stitched_images(folder, size=(256, 256)):
    low_light_images = []
    well_lit_images = []
    filenames = os.listdir(folder)
    
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename))
        
        if img is not None:
            h, w, _ = img.shape
            mid = w // 2
            low_light_img = img[:, :mid]
            well_lit_img = img[:, mid:]
            
            low_light_img = cv2.resize(low_light_img, size)
            well_lit_img = cv2.resize(well_lit_img, size)
            low_light_images.append(low_light_img)
            well_lit_images.append(well_lit_img)
    
    return low_light_images, well_lit_images, filenames

# Function to calculate image characteristics
def calculate_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    entropy = stats.entropy(np.histogram(gray, bins=256, range=(0, 256))[0])
    blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return mean_intensity, std_intensity, entropy, blurriness

# Directory
stitched_folder = 'combined_resized'  

# Load and preprocess images
low_light_images, well_lit_images, filenames = load_and_preprocess_stitched_images(stitched_folder)

# Extract features
low_light_features = [calculate_features(img) for img in low_light_images]
well_lit_features = [calculate_features(img) for img in well_lit_images]

low_light_mean, low_light_std, low_light_entropy, low_light_blur = zip(*low_light_features)
well_lit_mean, well_lit_std, well_lit_entropy, well_lit_blur = zip(*well_lit_features)

# Detect outliers using Z-score
def detect_outliers_zscore(data):
    z_scores = np.abs(stats.zscore(data))
    return np.where(z_scores > 3)[0]

outliers_low_light_mean = detect_outliers_zscore(low_light_mean)
outliers_low_light_std = detect_outliers_zscore(low_light_std)
outliers_low_light_entropy = detect_outliers_zscore(low_light_entropy)
outliers_low_light_blur = detect_outliers_zscore(low_light_blur)

outliers_well_lit_mean = detect_outliers_zscore(well_lit_mean)
outliers_well_lit_std = detect_outliers_zscore(well_lit_std)
outliers_well_lit_entropy = detect_outliers_zscore(well_lit_entropy)
outliers_well_lit_blur = detect_outliers_zscore(well_lit_blur)

all_outliers = set(outliers_low_light_mean).union(outliers_low_light_std, outliers_low_light_entropy, outliers_low_light_blur,
                                                  outliers_well_lit_mean, outliers_well_lit_std, outliers_well_lit_entropy, outliers_well_lit_blur)

# Convert outliers to standard Python integers for JSON serialization
all_outliers = [int(index) for index in all_outliers]

# Function to display images in a grid
def display_image_grid(images, title):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=20)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.axis('off')
        else:
            ax.axis('off')
    plt.savefig(f'{title}.png')
    plt.show()

# Function to overlay histograms with highlighted outliers
def plot_histograms_with_outliers(images, outliers, title):
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        if i in outliers:
            plt.plot(hist, alpha=0.7, color='red')
        else:
            plt.plot(hist, alpha=0.2, color='blue')
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig(f'{title}.png')
    plt.show()

# Scatter plot with highlighted outliers
def scatter_plot_with_outliers(x, y, outliers, xlabel, ylabel, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.scatter([x[i] for i in outliers], [y[i] for i in outliers], color='red', label='Outliers')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(f'{title}_scatter.png')
    plt.show()

# Plot histograms of pixel intensities with highlighted outliers
plot_histograms_with_outliers(low_light_images, all_outliers, "Histogram of Low Light Images with Outliers")
plot_histograms_with_outliers(well_lit_images, all_outliers, "Histogram of Well Lit Images with Outliers")

# Scatter plot of features with highlighted outliers
scatter_plot_with_outliers(low_light_mean, low_light_blur, all_outliers, 'Mean Intensity', 'Blurriness', 'Scatter Plot of Low Light Images')
scatter_plot_with_outliers(well_lit_mean, well_lit_blur, all_outliers, 'Mean Intensity', 'Blurriness', 'Scatter Plot of Well Lit Images')

# Annotated image grids for outliers
outlier_images = [low_light_images[i] for i in all_outliers if i < len(low_light_images)]
outlier_images += [well_lit_images[i] for i in all_outliers if i < len(well_lit_images)]

display_image_grid(outlier_images, "Detected Outliers")

# Collect names of images that are not outliers
cleaned_filenames = [filename for i, filename in enumerate(filenames) if i not in all_outliers]

# Save the cleaned filenames to a JSON file
output_json = 'cleaned_filenames.json'
with open(output_json, 'w') as f:
    json.dump(cleaned_filenames, f, indent=4)

print(f"Cleaned filenames saved to {output_json}")

# Save the outliers to a JSON file
output_json = 'outliers.json'
with open(output_json, 'w') as f:
    json.dump(list(all_outliers), f, indent=4)

print(f"Outliers saved to {output_json}")
