import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score,precision_score,accuracy_score
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict


# Author: Guni


# Ensure reproducibility in randomness
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size=128

transform = transforms.Compose([
    transforms.ToTensor(), # make sure pixel values are [0,1]
    transforms.Normalize((0.5,), (0.5,)) # normalize to [-1, 1]
    # helps stabling gradients
])

augmentated_transform = transforms.Compose([
    # apply augmentations on data to enhance diversity of the images, to help the model generelize and lower overfitting
    # we use them only for training, not for validation and testing because we obviously want to evaluate on realistic data
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),  # make sure pixel values are [0,1]
    transforms.Normalize((0.5,), (0.5,)) # normalize to [-1, 1]
    # helps stabling gradients
])

# create my own dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

dataset_path = 'spatial_envelope_256x256_static_8outdoorcategories'

# Get image paths and labels
image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
labels = [f.split('_')[0] for f in os.listdir(dataset_path)]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into training, validation, and testing sets (60 percent training, 20 percent validation, 20 percent testing)
# Using constant seed to ensure reproducibility in randomness (I found this one to work well)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, encoded_labels, test_size=0.4, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42
)

# Create datasets and loaders
train_dataset = CustomDataset(train_paths, train_labels, augmentated_transform)
val_dataset = CustomDataset(val_paths, val_labels, transform)
test_dataset = CustomDataset(test_paths, test_labels, transform)

# not shuffling val_loader,test_loader because evaluation needs to be persistent in order to compare with training data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# for question 3.a
def show_images_with_text(train_loader, label_encoder, num_examples=3):
    class_examples = defaultdict(list)
    
    # Group images by class
    for images, labels in train_loader:
        for img, label in zip(images, labels):
            if len(class_examples[label.item()]) < num_examples:
                class_examples[label.item()].append(img)
    
    num_classes = len(label_encoder.classes_)
    _, axes = plt.subplots(num_classes, num_examples, figsize=(num_examples * 4, num_classes * 4))
    
    for label, imgs in class_examples.items():
        class_name = label_encoder.inverse_transform([label])[0]
        for i, img in enumerate(imgs):
            if num_classes == 1:
                ax = axes[i]
            elif num_examples == 1:
                ax = axes[label]
            else:
                ax = axes[label, i]
            
            ax.imshow(img.permute(1, 2, 0)) 
            ax.axis("off")
            
            if i == 0:  
                ax.add_patch(Rectangle((0, 0), 256, 25, color="black", zorder=2))
                ax.text(
                    5, 12, class_name, 
                    fontsize=12, color="white", 
                    verticalalignment="center", 
                    zorder=3, weight='bold'
                )
    
    plt.tight_layout()
    plt.show()

#show_images_with_text(train_loader, label_encoder, num_examples=3)
#show_images_with_text(test_loader, label_encoder, num_examples=3)

# Residual block with skip connection
# These blocks help in case a deep learning network is too deep, some layers are useless and might actually cause the network to be less good (might overfit)
# lets suppose the output of a layer is y1, so the output of the residual block is y=y1+x (x is the input for the network)
# this basically allows the network to learn the identity function when it needs to, and this way to neglect the excessing layers
# That was the first purpose researches thought about making those blocks, but in retrospect these blocks also solve another problem
# which is the zeroing gradient - as the gradient of the cost function is basically a big product of gradients, it is enough that
# only one of them will be zeroed (for example by ReLU activation) so that it will all get zeroed.
# This way, with the residual block, because we add the identity function the gradient won't get zeroed
# Also, a residual block has to be atleast with size of 2 analogically to linear layers where 2 of them without non-linear activation in between
# are basically just like 1 layer

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution layer with optional downsampling (with stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        # Batch normalization layer to stabilize learning by normalizing activations
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Non linearity 
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution layer without downsampling
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Shortcut connection
        shortcut = self.shortcut(x)
        
        # Main path through two convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # Combine shortcut with main path
        x += shortcut
        
        # Final activation
        return self.relu(x)

# This model implements residual connections for deeper architectures,
# batch normalization for stable training, and pooling for dimension reduction.

class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        # Initial convolution to extract basic features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(64)
        
        # Non linearity
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling to reduce dimensions
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers for feature learning
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # reduce dimensions to 1x1 with average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(512, num_classes)

    # Generates sequence of Residual Blocks.
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # feature learning through residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling to prepare for classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, features)
        
        # Classification
        return self.fc(x)

def reset_param(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

model = CNN(num_classes=len(label_encoder.classes_)).to(device)

reset_param(model)

# Loss function (equivilent to applying logsoftmax and then NLLLoss)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# adam allows adaptive learning rate - parameters with much influence will be updated slowly and the opposite
# added l2 regularization to prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
# adjust the learning rate of the optimizer during training to help
# the model converge more effectively by controlling the step size during parameter updates
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

epochs = 20
train_loss_history, val_loss_history = [], []
early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0

# Training and validation loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss_history.append(running_loss / len(train_loader))
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item()
    val_loss_history.append(val_loss / len(val_loader))
    
    # Adjust the learning rate based on validation loss
    scheduler.step(val_loss)  

    # Print the learning rate after each epoch (using get_last_lr)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, Train Loss: {train_loss_history[-1]:.4f}, Val Loss: {val_loss_history[-1]:.4f}, Learning Rate: {current_lr:.6f}")

# Show training and validation loss
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Testing loop:
model.eval()
all_labels, all_outputs = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

all_outputs = torch.cat(all_outputs)
all_labels = torch.cat(all_labels)

# Compute ROC curves 
one_hot_labels = torch.nn.functional.one_hot(all_labels, num_classes=len(label_encoder.classes_))
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(one_hot_labels[:, i].numpy(), all_outputs[:, i].numpy())
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Compute Precision-Recall curves 
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(label_encoder.classes_):
    # Compute Precision-Recall curve for each class
    precision, recall, _ = precision_recall_curve(one_hot_labels[:, i], all_outputs[:, i])
    # Compute Average Precision for each class
    avg_precision = average_precision_score(one_hot_labels[:, i], all_outputs[:, i])
    # show the Precision-Recall curve for the class
    plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.2f})')

plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.show()

# Compute model precision
_, predicted = torch.max(all_outputs, 1)
precision = precision_score(all_labels.numpy(), predicted.numpy(), average='macro')
print(f"Test Precision (Macro Average): {precision:.4f}")

# Compute model accuracy
_, predicted = torch.max(all_outputs, 1)
accuracy = accuracy_score(all_labels.numpy(), predicted.numpy())
print(f"Test Accuracy: {accuracy:.4f}")
