#Import the necessary libraries 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import imageio.v3 as iio
import os

def ver_flip(img, save_file_path) -> None:
    flipped_img = cv2.flip(img, 0)
    flipped_img_rgb = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, flipped_img_rgb)

def hol_flip(img, save_file_path) -> None:
    flipped_img = cv2.flip(img, 1)
    flipped_img_rgb = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, flipped_img_rgb)

def ver_hol_flip(img, save_file_path) -> None:
    flipped_img = cv2.flip(img, 0)
    flipped_img = cv2.flip(img, 1)
    flipped_img_rgb = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, flipped_img_rgb)

def rot_90(img, save_file_path) -> None:
    rows, cols = img.shape[:2]
    Cx , Cy = rows, cols
    M = cv2.getRotationMatrix2D((Cy//2, Cx//2),90 ,1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))
    rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, rotated_img_rgb)

def rot_n90(img, save_file_path) -> None:
    rows, cols = img.shape[:2]
    Cx , Cy = rows, cols
    M = cv2.getRotationMatrix2D((Cy//2, Cx//2), -90 ,1)
    rotated_img = cv2.warpAffine(img, M, (cols, rows))
    rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, rotated_img_rgb)

def blur_gauss(img, save_file_path) -> None:
    blured_img = cv2.blur(img, (5,5))
    blured_img_rgb = cv2.cvtColor(blured_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, blured_img_rgb)

def extract_and_tile_patches(img, save_file_path, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)):
    # 画像にパディングを追加
    padded_image = cv2.copyMakeBorder(img, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_CONSTANT,
                                      value=0)

    # パディングを加えた画像の次元を取得
    padded_height, padded_width = padded_image.shape[:2]

    # パッチを抽出する
    patches = []
    for y in range(0, padded_height - kernel_size[1] + 1, stride[1]):
        for x in range(0, padded_width - kernel_size[0] + 1, stride[0]):
            patch = padded_image[y:y + kernel_size[1], x:x + kernel_size[0]]
            patches.append(patch)

    # パッチの数を計算
    num_patches_x = (padded_width - kernel_size[0]) // stride[0] + 1
    num_patches_y = (padded_height - kernel_size[1]) // stride[1] + 1

    # 新しい画像のサイズを計算
    tiled_image_height = num_patches_y * kernel_size[1]
    tiled_image_width = num_patches_x * kernel_size[0]

    # 新しい画像を作成
    tiled_img = np.zeros((tiled_image_height, tiled_image_width, 3), dtype=np.uint8)

    # パッチを新しい画像に配置
    patch_idx = 0
    for y in range(0, tiled_image_height, kernel_size[1]):
        for x in range(0, tiled_image_width, kernel_size[0]):
            tiled_img[y:y + kernel_size[1], x:x + kernel_size[0]] = patches[patch_idx]
            patch_idx += 1

    tiled_img_rgb = cv2.cvtColor(tiled_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, tiled_img_rgb)



directory = 'low-light-enhancement/data_augm'

for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        file_path = os.path.join(directory, filename)
        base_name = os.path.basename(file_path)  # Get the base name
        file_name_simple, _ = os.path.splitext(base_name)  # Split the file name and extension

        ver_flip_path = os.path.join(directory,'ver', filename)
        hol_flip_path = os.path.join(directory,'hol', filename)
        ver_hol_flip_path = os.path.join(directory, 'verhol', filename)
        rot_90_path = os.path.join(directory, 'rot90', filename)
        rot_n90_path = os.path.join(directory, 'rotn90', filename)
        blur_path = os.path.join(directory, 'blur', filename)
        tiled_patch_path = os.path.join(directory, 'tiled', filename)

        # Process the image file
        image = iio.imread(file_path)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        ver_flip(image, ver_flip_path)
        hol_flip(image, hol_flip_path)
        ver_hol_flip(image, ver_hol_flip_path)    
        rot_90(image, rot_90_path)
        rot_n90(image, rot_n90_path)
        blur_gauss(image, blur_path)
        extract_and_tile_patches(image, tiled_patch_path, kernel_size=(3,3), stride = (1,1), padding = (0,0))

