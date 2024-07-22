#Import the necessary libraries 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import imageio.v3 as iio

def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
        return flat_list

def color2gray(img):
    if img.ndim == 3:  # Check if the image has color channels
        gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    else:
        gray_img = img  # If the image is already grayscale
    return gray_img

def find_nearest(ref_array, value):
    idx = (np.abs(np.array(ref_array) - value)).argmin()
    return idx

def clahe_hist(img_v):
    img_v = img_hsv[:,:,2].astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img_v)
    return cl1

def intensity_factoring(img_v, intensity_factor):
    increased_value_channel = np.clip(img_v * intensity_factor, 0, 255).astype(np.uint8)
    return increased_value_channel

def retinex_model(img):
    c_max = [np.max(img[:,:,i]) for i in range(img.shape[2])]
    ratio = [c_max[1]/c_max[i] for i in range(img.shape[2])]
    print(ratio)
    new_img = np.stack((ratio[0]*img[:,:,0], ratio[1]*img[:,:,1], ratio[2]*img[:,:,2]), axis=-1)
    return ratio, new_img


# Load the image 
image = iio.imread('low_light_enhancement/limestone_cave.jpg') 

#Plot the original image 
plt.subplot(2, 4, 1) 
plt.title("Original") 
plt.imshow(image) 

img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

shift_clh = clahe_hist(img_hsv[:,:,2])
shifting_result_clh = img_hsv
shifting_result_clh[:,:,2] = shift_clh

intensity_factor = 5.0
shift_fac = intensity_factoring(img_hsv[:,:,2], intensity_factor)
shifting_result_fac = img_hsv
shifting_result_fac[:,:,2] = shift_fac

ratio, shifting_result_rei = retinex_model(image)


#Plot the contrast image 
plt.subplot(2, 4, 2) 
plt.title("Brightness shift CLAHE") 
plt.imshow(cv2.cvtColor(shifting_result_clh, cv2.COLOR_HSV2RGB)) 

plt.subplot(2, 4, 3) 
plt.title("Brightness shift factoring") 
plt.imshow(cv2.cvtColor(shifting_result_fac, cv2.COLOR_HSV2RGB)) 

plt.subplot(2, 4, 4) 
plt.title("Brightness shift Retinex") 
plt.imshow(cv2.cvtColor(shifting_result_fac, cv2.COLOR_HSV2RGB)) 

# Plot the histograms of intensities
plt.subplot(2, 4, 5)
plt.title('Histogram')
plt.hist(flatten_extend(img_hsv[:,:,2]), bins=np.arange(0,255,1))

plt.subplot(2, 4, 6)
plt.title('Histogram')
plt.hist(flatten_extend(shifting_result_clh[:,:,2]), bins=np.arange(0,255,1))

plt.subplot(2, 4, 7)
plt.title('Histogram')
plt.hist(flatten_extend(shifting_result_fac[:,:,2]), bins=np.arange(0,255,1))

plt.subplot(2, 4, 8)
plt.title('Histogram')
plt.hist(flatten_extend(shifting_result_rei[:,:,2]), bins=np.arange(0,255,1))

plt.show()
