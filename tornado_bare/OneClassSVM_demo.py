from sklearn.svm import OneClassSVM
import pandas as pd
from skimage import io, color, transform
from skimage.feature import hog
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os

# New Image and label collection method
# Load the CSV file
csv_file = "./Images/labels.csv"
data = pd.read_csv(csv_file)

# Parameters
img_size = (128, 128)

# Load images and labels
def load_data(data):
    images = []
    labels = []
    for _, row in data.iterrows():
        image = io.imread(row['Filename'])
        image = transform.resize(image, img_size, anti_aliasing=True)  # Resize image
        images.append(image)
        labels.append(row['Label'])
    return images, labels

# Split data into training and test sets
train_data = data[data['Filename'].str.contains("Dogs")] # Filepaths containing "Dogs"
test_data = data[data['Filename'].str.contains("Test")] # Filepaths containing "Test"

train_images, train_labels = load_data(train_data)
test_images, test_labels = load_data(test_data)

def augment_images(images, labels):
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        augmented_images.append(img)  # Original image
        augmented_labels.append(label)

        # Rotations
        for angle in [90, 180, 270]:
            rotated_img = transform.rotate(img, angle)
            augmented_images.append(rotated_img)
            augmented_labels.append(label)

        # Flipping
        flipped_img = img[:, ::-1]
        augmented_images.append(flipped_img)
        augmented_labels.append(label)

        # Add noise
        noisy_img = random_noise(img, mode='gaussian', var=0.01)
        augmented_images.append(noisy_img)
        augmented_labels.append(label)

        # Adjust brightness
        brighter_img = adjust_gamma(img, gamma=0.5)
        darker_img = adjust_gamma(img, gamma=2.0)
        augmented_images.extend([brighter_img, darker_img])
        augmented_labels.extend([label, label])

    return augmented_images, augmented_labels


# Apply PCA to reduce dimensionality
def apply_pca(X_train, X_test, explained_variance=0.95):
    pca = PCA(n_components=explained_variance)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train)    # Fit and transform on training data
    X_test_pca = pca.transform(X_test)          # Transform test data
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Reduced number of features: {X_train_pca.shape[1]}")
    return X_train_pca, X_test_pca


def create_dataset_with_hog(img_list, label_list=None): # code from Chat-GPT
    pixels_per_cell = (4, 4) # higher values capture more global patterns (i.e. (16,16))
    cells_per_block = (2, 2) # higher values less sensitive to fine-grained features
    orientations = 9

    # Convert to grayscale and extract HOG features
    h, w = 128, 128
    g_img = [color.rgb2gray(x) for x in img_list]  # Convert to grayscale
    resize_img = [transform.resize(img, (h, w), anti_aliasing=True) for img in g_img]  # Resize images

    # Extract HOG features for each image
    hog_features = [hog(img, 
                         orientations=orientations, 
                         pixels_per_cell=pixels_per_cell, 
                         cells_per_block=cells_per_block, 
                         block_norm='L2-Hys') 
                    for img in resize_img]

    # Convert HOG features to numpy array
    X = np.array(hog_features)

    return X


augmented_images, augmented_labels = augment_images(train_images, train_labels)
X_train = create_dataset_with_hog(augmented_images)
X_test = create_dataset_with_hog(test_images)

X_train_pca, X_test_pca = apply_pca(X_train, X_test)

# Print data shapes
# print("Shape of X_train:", X_train_pca.shape)
# print("Shape of X_test:", X_test_pca.shape)

# Train a OneClassSVM
ocsvm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)  # nu=0.05 for 5% of outliers
ocsvm.fit(X_train)  # Only fit on the "normal" data (inliers)

print("Training complete!")

# Predict on test set
# y_pred = ocsvm.predict(X_test_pca)  # 1 for inliers, -1 for outliers
y_pred = ocsvm.predict(X_test)  # 1 for inliers, -1 for outliers

print("Predictions:", y_pred)

# Count the number of anomalies detected
anomalies = np.sum(y_pred == -1)
print(f"Number of anomalies detected: {anomalies}")

# Optional: You can also print or visualize the results in some way
# For example, visualizing some anomalies in test set images:
plt.imshow(test_images[0])
plt.title(f"Prediction: {'Anomaly' if y_pred[0] == -1 else 'Normal'}")
plt.show()