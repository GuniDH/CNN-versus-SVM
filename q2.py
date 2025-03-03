import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, normalize, label_binarize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, confusion_matrix, average_precision_score, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Author: Guni 


""" Returns arrays of loaded images and their class labels """
def load_dataset(dataset_path):
    images, labels = [], []

    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        img = Image.open(img_path).convert("RGB") # Read the image 
        if img is None:
            continue 
        images.append(img)
        label = img_name.split('_')[0] # Extract the class label from the filename
        labels.append(label)

    return images, labels


""" Returns list which contains features for each image (each image is represented by a list of features found by ImageNet VGG network) """
def compute_features(images):
    # Check if CUDA is available and use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pre-trained vgg-16 model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    
    # Remove the classifier (fully connected layers) to get the feature map from the last conv layer
    feature_extractor = nn.Sequential(*list(model.children())[:-2]).to(device)
    
    # Set the feature extractor to evaluation mode
    feature_extractor.eval()
    
    # Image transformation (maintain original size)
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize( # Normalize to ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    features = [] # Each element is the list of features for each image

    for img in images:
        with torch.no_grad(): # Model is already trained, this is the forward pass so no need to compute gradient
            input_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension and move to GPU
            # Forward pass through the feature extractor
            feat_map = feature_extractor(input_tensor).cpu() # Move the features back to CPU 
            feat_map = feat_map.flatten(start_dim=2).transpose(1, 2).reshape(-1, 512) # Reshape into 64 vectors of size 512 for later processing (from shape of 1*512*8*8 to 64*512)
            features.append(feat_map)  

    return features



""" Perform PCA and KMEANS then map the labeled cluster for each feature of an image """
def quantizate_vectors(features, n_clusters):
    # Flatten all features of all images into a single list so we could perform KMEANS and PCA
    flattened_features = [feature for img_features in features for feature in img_features]

    # PCA to reduce data with high covariance and thus lower dimensions of the data
    # Select top components and thus reduce each feature by 8 times from size 512 to 64
    # I Choose constant random seed for reproducibility of SVD solving (algorithm might use randomized svd solver)
    pca = PCA(n_components=64, random_state=42)
    reduced_features = pca.fit_transform(flattened_features)

    # Perform clustering for all features (at the end each features will have its labeled cluster-the closest cluster for the specific features)
    # I Choose constant random seed to keep reproducibility of this cluster initilization which i found to work well
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_features) # Array in which each element is the index of cluster each features was assigned to
    codewords_dictionary = kmeans.cluster_centers_
    
    # Transform labels list to match the shape of features so we could create histograms for each image
    i = 0
    reshaped_labels = []
    for img_features in features:
        img_labels = labels[i:i + len(img_features)]
        reshaped_labels.append(img_labels)
        i += len(img_features)

    return reshaped_labels, codewords_dictionary


""" Create histograms as representation for each image (how many features were labeled for each cluster) """
def create_histograms(labeled_features, codewords_dictionary):
    histograms = []
    for img_labeled_features in labeled_features:
        # Create histogram for each image
        histogram, _ = np.histogram(img_labeled_features, bins=np.arange(len(codewords_dictionary) + 1))
        histograms.append(histogram)

    # Ensures the histograms have consistent magnitudes because each histogram is converted into a unit vector
    # while making sure the comparison between histograms will be based on the relative distribution of the features
    histograms = normalize(histograms, norm='l2') 
    return histograms


""" Train validate and test data with SVM """
def train_validate_test(histograms, labels):
    # Convert to np array as it's more suitable for the next calculations
    X = np.array(histograms)

    # Convert string labels to integer labels using LabelEncoder
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_  # Store class names

    # I Choose constant random seed for reproducibility of data splitting
    # Split data into 60% training and 40% temporary (validation + test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
    # Split the temporary data into 50% validation and 50% test (20% each)
    X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Optimize SVM hyperparameters using GridSearchCV
    svm = SVC(probability=True)
    param_grid = {
        'C': [0.1, 1, 10], # Regularization parameter (to reduce overfitting the hyperplane for extreme data points (histograms))
        'gamma': [0.01, 0.1, 1], # Defines the influence of a single histogram
        'kernel': ['rbf'] # Defines the type of hyperplane used to separate the data
    }
    # For each combination of C and gamma, the data is split into 3 parts for training and validating (due to cv=3)
    grid_search = GridSearchCV(svm, param_grid, cv=3)
    grid_search.fit(X_train, Y_train)
    best_svm = grid_search.best_estimator_
    print(f"Best SVM Parameters: {grid_search.best_params_}")

    # Validate the model
    Y_val_pred = best_svm.predict(X_val)
    Y_val_prob = best_svm.predict_proba(X_val)

    # Test the model
    Y_test_pred = best_svm.predict(X_test)
    Y_test_prob = best_svm.predict_proba(X_test)

    # Binarize labels in a one-vs-all fashion (roc_curve expects binary input)
    # (It's actually converted to unary base)
    Y_val_bin = label_binarize(Y_val, classes=np.unique(Y_val))
    Y_test_bin = label_binarize(Y_test, classes=np.unique(Y_test))

    fpr_val, tpr_val, roc_auc_val = {}, {}, {}
    fpr_test, tpr_test, roc_auc_test = {}, {}, {}

    for i in range(Y_val_bin.shape[1]):
        fpr_val[i], tpr_val[i], _ = roc_curve(Y_val_bin[:, i], Y_val_prob[:, i])
        fpr_test[i], tpr_test[i], _ = roc_curve(Y_test_bin[:, i], Y_test_prob[:, i])
        roc_auc_val[i] = auc(fpr_val[i], tpr_val[i]) 
        roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])

    # ROC curve shows the trade offs between the True Positive Rate and the False Positive Rate

    plt.figure(figsize=(12, 6))

    # ROC Curve for validation 
    ax1 = plt.subplot(1, 2, 1)  
    for i in range(Y_val_bin.shape[1]):
        ax1.plot(fpr_val[i], tpr_val[i], label=f'{class_names[i]} - Validation (AUC = {roc_auc_val[i]:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic - Validation')
    ax1.legend(loc='lower right')

    # ROC Curve for test
    ax2 = plt.subplot(1, 2, 2) 
    for i in range(Y_test_bin.shape[1]):
        ax2.plot(fpr_test[i], tpr_test[i], label=f'{class_names[i]} - Test (AUC = {roc_auc_test[i]:.2f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic - Test')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    roc_auc_val_macro = roc_auc_score(Y_val_bin, Y_val_prob, average='macro', multi_class='ovr')
    roc_auc_test_macro = roc_auc_score(Y_test_bin, Y_test_prob, average='macro', multi_class='ovr')

    print(f"Validation AUC (One-vs-Rest): {roc_auc_val_macro:.2f}")
    print(f"Test AUC (One-vs-Rest): {roc_auc_test_macro:.2f}")

    # Precision-Recall curve for each class (One-vs-Rest)
    # Precision tells us how many of the positive predictions made by the model were actually correct
    # High precision means that when the model predicts a positive class then it's likely to be correct
    # Recall tells us how many of the actual positive instances were correctly identified by the model (Sensitivity)
    # High recall means that the model is able to correctly identify most of the actual positive instances.

    # Precision-Recall curve for each class (One-vs-Rest)
    plt.figure(figsize=(12, 6))

    # Precision-recall curve for validation
    ax1 = plt.subplot(1, 2, 1)
    for i in range(Y_val_bin.shape[1]):
        precision_val, recall_val, _ = precision_recall_curve(Y_val_bin[:, i], Y_val_prob[:, i])
        ap_val = average_precision_score(Y_val_bin[:, i], Y_val_prob[:, i])  
        ax1.plot(recall_val, precision_val, label=f'{class_names[i]} - Validation (AP = {ap_val:.2f})')  
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve - Validation')
    ax1.legend(loc='best')

    # Precision-recall curve for test
    ax2 = plt.subplot(1, 2, 2)
    for i in range(Y_test_bin.shape[1]):
        precision_test, recall_test, _ = precision_recall_curve(Y_test_bin[:, i], Y_test_prob[:, i])
        ap_test = average_precision_score(Y_test_bin[:, i], Y_test_prob[:, i]) 
        ax2.plot(recall_test, precision_test, label=f'{class_names[i]} - Test (AP = {ap_test:.2f})')  
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve - Test')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm_val = confusion_matrix(Y_val, Y_val_pred)
    cm_test = confusion_matrix(Y_test, Y_test_pred)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm_val).plot(ax=axes[0], cmap='Purples')
    axes[0].set_title('Validation Confusion Matrix')
    ConfusionMatrixDisplay(confusion_matrix=cm_test).plot(ax=axes[1], cmap='Purples')
    axes[1].set_title('Test Confusion Matrix')
    plt.show()


def main():
    
    dataset_path = 'spatial_envelope_256x256_static_8outdoorcategories'
    print("Loading dataset...")
    images, labels = load_dataset(dataset_path)
    
    print("Computing features...")
    features = compute_features(images)
    
    print("Quantizing vectors...")
    labeled_features, codewords_dictionary = quantizate_vectors(features, n_clusters=200)
    
    print("Creating histograms...")
    histograms = create_histograms(labeled_features, codewords_dictionary)
    
    print("Training and evaluating SVM...")
    train_validate_test(histograms, labels)


if __name__ == '__main__':
    main()
