# DL-Forest-Semantic-Segmentation
# 📝 Preprocessing Notebook

## 📌 Overview
This repository contains the notebook **`Prep.ipynb`**, which is used to **prepare and preprocess satellite images and masks** for semantic segmentation tasks.  
All steps in this notebook are dedicated to cleaning, augmenting, and organizing the dataset before training a deep learning model.

---

## ⚙️ Steps in `Prep.ipynb`

1. **Setup and Imports**
   - Import core libraries: `torch`, `torchvision`, `albumentations`, `numpy`, `pandas`, `os`, and `random`.
   - Configure reproducibility by fixing random seeds for Python, NumPy, and PyTorch.

2. **Define Image and Mask Parameters**
   - Specify image size: **160 × 160 pixels**, with 3 channels (RGB).  
   - Specify mask size: **160 × 160 pixels**, with 1 channel (grayscale).  
   - Define constants for batch size and learning rate to be used later.

3. **Data Augmentation**
   - Apply augmentation strategies using **Albumentations**:
     - Resize all images and masks to the fixed dimensions.  
     - Random horizontal and vertical flips to increase variability.  
     - Random 90-degree rotations for additional robustness.  

4. **Metadata Preparation**
   - Load a CSV file containing filenames of images and their corresponding masks.  
   - Build **absolute file paths** for both image and mask columns.  
   - Inspect the dataset structure using `pandas.head()` to confirm correct loading.

5. **Dataset and Dataloader Construction**
   - Implement a custom function to load image–mask pairs into PyTorch datasets.  
   - Split the dataset into **training and testing sets** with a user-defined ratio (default 90% training / 10% testing).  
   - Use PyTorch `DataLoader` to:
     - Batch and shuffle the dataset.  
     - Optionally repeat the dataset multiple times for augmentation.  
   - Create ready-to-use **`train_loader`** and **`test_loader`** objects.

---

## 📂 File
- `Prep.ipynb` – Jupyter Notebook containing all preprocessing code.
