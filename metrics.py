import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd
import matplotlib.pyplot as plt

def calculate_mrae(pred, gt):
    epsilon = 0
    return np.mean(np.abs(pred - gt) / (np.abs(gt) + epsilon))

def calculate_rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def calculate_psnr(pred, gt):
    return compare_psnr(gt, pred)

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[filename] = img
    return images

def evaluate_model(gt_folder, pred_folder):
    gt_images = load_images_from_folder(gt_folder)
    pred_images = load_images_from_folder(pred_folder)

    results = []

    for filename in gt_images:
        if filename in pred_images:
            gt_img = gt_images[filename]
            pred_img = pred_images[filename]

            if gt_img.shape != pred_img.shape:
                print(f"Skipping {filename}: dimension mismatch.")
                continue

            mrae = calculate_mrae(pred_img, gt_img)
            rmse = calculate_rmse(pred_img, gt_img)
            psnr = calculate_psnr(pred_img, gt_img)

            results.append({
                "Image": filename,
                "MRAE": mrae,
                "RMSE": rmse,
                "PSNR": psnr
            })
        else:
            print(f"Skipping {filename}: no corresponding prediction found.")

    df = pd.DataFrame(results)
    avg_metrics = df.mean(numeric_only=True).to_dict()

    return df, avg_metrics

def plot_metrics(all_results, metric_name):
    images = all_results[list(all_results.keys())[0]]["Individual Metrics"]["Image"]
    x = np.arange(len(images))

    width = 0.2  # width of the bars
    fig, ax = plt.subplots()

    for i, (model_name, result) in enumerate(all_results.items()):
        metric_values = result["Individual Metrics"][metric_name]
        ax.bar(x + i * width, metric_values, width, label=model_name)

    ax.set_xlabel('Images')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} for Different Models')
    ax.set_xticks(x + width / len(all_results))
    ax.set_xticklabels(images, rotation=90)
    ax.legend()
    if metric_name == "MRAE":
        plt.ylim(0,3)
    elif metric_name == "RMSE":
        plt.ylim(0, 15)
    elif metric_name == "PSNR":
        plt.ylim(0, 22)
    plt.savefig(f'{metric_name}.png')
    plt.show()

def main():
    models = {
        "cbam": "cbam/Predicted_Image",
        "pix2pix": "pix2pix-new/Predicted_Image",
        "mirnet": "mirnet/Predicted_Image",
        "unet": "unet/Predicted_Image"
    }
    # gt_folder = 'unet/Ground_Truth'

    all_results = {}
    for model_name, pred_folder in models.items():
        if model_name == "cbam":
            gt_folder = 'cbam/Ground_Truth'
        if model_name == "pix2pix":
            gt_folder = 'pix2pix-new/Ground_Truth'
        if model_name == "mirnet":
            gt_folder = 'mirnet/Ground_Truth'
        if model_name == "unet":
            gt_folder = 'unet/Ground_Truth'
        df, avg_metrics = evaluate_model(gt_folder, pred_folder)
        all_results[model_name] = {
            "Individual Metrics": df,
            "Average Metrics": avg_metrics
        }
        print(f"\nModel: {model_name}")
        print(df)
        print(f"Average Metrics: {avg_metrics}")

    # Plot metrics
    for metric in ["MRAE", "RMSE", "PSNR"]:
        plot_metrics(all_results, metric)

if __name__ == "__main__":
    main()
