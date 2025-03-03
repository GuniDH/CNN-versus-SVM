import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, normalize, label_binarize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, confusion_matrix, average_precision_score, ConfusionMatrixDisplay


# Author: Guni Deyo Haness 215615519
# Question 1 for maman 12 in computer vision course


""" Returns arrays of loaded images and their class labels """
def load_dataset(dataset_path):
    images, labels = [], []

    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) # Read the image in grayscale
        if img is None:
            continue 
        images.append(img)
        label = img_name.split('_')[0] # Extract the class label from the filename
        labels.append(label)

    return images, labels


""" Returns list which contains descriptors for each image (each image is represented by a list of descriptors for the keypoints found by SIFT) """
def compute_descriptors(images):
    # Keep up to top 500 significant keypoints (and their descriptors)

    # Keypoints with contrast lower than contrastThreshold will be filtered out (low contrast might indicate noise=>uninformative points)

    # Use edge threshold to filter more edge keypoints retaining more balanced edges (such as corner)
    # Keypoints on edges such as long straight ones aren't as informative because they are similar

    sift = cv.SIFT_create(nfeatures=500, contrastThreshold=0.04, edgeThreshold=10)
    descriptors = [] # Each element is the list of descriptors for each image

    for img in images:
        _, img_descriptors = sift.detectAndCompute(img, None) # Get all descriptors for all keypoints of an image
        # Dimensions are number_of_keypoints * 128. Each descriptor is a 128 feature vector because each keypoint is 
        # split to 4*4 grid where each component is an 8 bin histogram of gradient orientations.
        # This way the histograms capture the distribution of edge directions in each component.
        if img_descriptors is not None:
            descriptors.append(img_descriptors)
    
    return descriptors


""" Perform PCA and KMEANS then map the labeled cluster for each descriptor of an image """
def quantizate_vectors(descriptors, n_clusters):
    # Flatten all descriptors of all images into a single list so we could perform KMEANS and PCA
    flattened_descriptors = [descriptor for img_descriptors in descriptors for descriptor in img_descriptors]

    # PCA to reduce data with high covariance and thus lower dimensions of the data
    # Select top components and thus reduce each SIFT descriptor by half from size 128 to 64
    # I Choose constant random seed for reproducibility of SVD solving (algorithm might use randomized svd solver)
    pca = PCA(n_components=64, random_state=42)
    reduced_descriptors = pca.fit_transform(flattened_descriptors)

    # Perform clustering for all descriptors (at the end each descriptor will have its labeled cluster-the closest cluster for the specific descriptor)
    # I Choose constant random seed to keep reproducibility of this cluster initilization which i found to work well
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_descriptors) # Array in which each element is the index of cluster each descriptor was assigned to
    codewords_dictionary = kmeans.cluster_centers_
    
    # Transform labels list to match the shape of descriptors so we could create histograms for each image
    i = 0
    reshaped_labels = []
    for img_descriptors in descriptors:
        img_labels = labels[i:i + len(img_descriptors)]
        reshaped_labels.append(img_labels)
        i += len(img_descriptors)

    return reshaped_labels, codewords_dictionary


""" Create histograms as representation for each image (how many features were labeled for each cluster) """
def create_histograms(labeled_descriptors, codewords_dictionary):
    histograms = []
    for img_labeled_descriptors in labeled_descriptors:
        # Create histogram for each image
        histogram, _ = np.histogram(img_labeled_descriptors, bins=np.arange(len(codewords_dictionary) + 1))
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
    
    print("Computing descriptors...")
    descriptors = compute_descriptors(images)
    
    print("Quantizing vectors...")
    labeled_descriptors, codewords_dictionary = quantizate_vectors(descriptors, n_clusters=200)
    
    print("Creating histograms...")
    histograms = create_histograms(labeled_descriptors, codewords_dictionary)
    
    print("Training and evaluating SVM...")
    train_validate_test(histograms, labels)


if __name__ == '__main__':
    main()
