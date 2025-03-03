# Image Classification using Classic Machine Learning versus CNN by Guni

## Overview

This project focuses on **image classification** using two different approaches:
1. **Classical Machine Learning** (Feature extraction with **SIFT** - q1.py or **VGG** - q2.py, KMeans clustering, and SVM classification by keywords histograms)
2. **Deep Learning** (convolutional neural network - q3.py)


## Structure

### `q1.py`
- Implements SVM classification with BoW representation such that features are found with **SIFT**,
- then KMEANS clustering is done on the whole features found in all the images and each image is represented
- by the histogram in which is i-th bin is how many features of the image where assigned to the i-th cluster during KMEANS.
- Evaluates model performance using **ROC-AUC, Precision-Recall, and Confusion Matrices**.

### `q2.py`
- Implements SVM classification with BoW representation such that features are found with **VGG**,
- then KMEANS clustering is done on the whole features found in all the images and each image is represented
- by the histogram in which is i-th bin is how many features of the image where assigned to the i-th cluster during KMEANS.
- Evaluates model performance using **ROC-AUC, Precision-Recall, and Confusion Matrices**.

### `q3.py`
- Implements a **Convolutional neural network**.
- Defines a **custom dataset loader** and training pipeline.
- Evaluates using **Accuracy, Precision, and AUC metrics**.

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

