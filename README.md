# Image Classification using Classic Machine Learning versus CNN by Guni

## Overview

This project focuses on **image classification** using two different approaches:
1. **Classical Machine Learning** (Feature extraction with **SIFT** - q1.py or **VGG** - q2.py, KMeans clustering, and SVM classification by keywords histograms)
2. **Deep Learning** (convolutional neural network - q3.py)


## Structure

### `q1.py`
- Implements **feature extraction using SIFT**.
- Extracts keypoints and descriptors from images.
- Applies **KMeans clustering** to group similar features and create a visual vocabulary.

### `q2.py`
- Implements **image classification using SVM**.
- Uses the **Bag of Visual Words (BoVW)** representation created from SIFT features.
- Applies **GridSearchCV** for hyperparameter tuning.
- Evaluates model performance using **ROC-AUC, Precision-Recall, and Confusion Matrices**.

### `q3.py`
- Implements a **Deep Learning model using PyTorch**.
- Defines a **custom dataset loader** and training pipeline.
- Uses **CNNs (Convolutional Neural Networks) for classification**.
- Evaluates using **Accuracy, Precision, and AUC metrics**.

## How It Works

### **Feature Extraction with SIFT (q1.py)**
1. Load each image and apply **SIFT** to extract keypoints and descriptors.
2. Use **KMeans clustering** to create a vocabulary of visual words.
3. Represent each image as a **histogram of visual words**.

### **Classification with SVM (q2.py)**
1. Train an **SVM model** using the BoVW representation.
2. Use **GridSearchCV** to find the best hyperparameters.
3. Evaluate using **Confusion Matrices, ROC-AUC, and Precision-Recall curves**.

### **Deep Learning with PyTorch (q3.py)**
1. Load the dataset using a **custom PyTorch Dataset class**.
2. Train a **CNN model** to classify images.
3. Evaluate performance using **AUC, Accuracy, and Precision**.

## Installation

Ensure you have Python installed and install the required dependencies:
```sh
pip install numpy matplotlib scikit-learn opencv-python torch torchvision
```

## Running the Program

1. **Clone the Repository**
   ```sh
   git clone https://github.com/GuniDH/CNN-versus-SVM.git
   cd CNN-versus-SVM
  
   ```
2. **Run the Feature Extraction & Clustering**
   ```sh
   python q1.py
   ```
3. **Train & Evaluate the SVM Model**
   ```sh
   python q2.py
   ```
4. **Train & Evaluate the Deep Learning Model**
   ```sh
   python q3.py
   ```


## License

This project is licensed under the **MIT License**.

---
### Author
**Guni**

