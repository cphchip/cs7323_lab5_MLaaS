#===================================================================================================
#   From ChatGPT - train and predict a dog in images
#
# This section of code was generated with the assistance of ChatGPT, an AI language model by OpenAI.
# Date: 11/18/24
# Source: OpenAI's ChatGPT (https://openai.com/chatgpt)
# Modifications: added remove of ".DS_Store" folders created by MacOs. 
#                Also added check to only read in "".jpg" files 
#
#
# TODO:  In Documentation or a README File
# If you're documenting the project or including external acknowledgments, you can add a note like this:
#
# Acknowledgment:
# Portions of this code were generated with the assistance of ChatGPT, an AI language model developed by OpenAI. 
# The generated code was reviewed, modified, and integrated into the project to meet specific requirements.
#--------------------------------------------------------------------------------------------------
import os
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib




def load_images_from_folder(folder_path, image_size=(64, 64)):
    """
    Load images from folder and return as a list of numpy arrays with labels.
    
    Args:
    - folder_path (str): Path to the data folder (e.g., Train or Test).
    - image_size (tuple): Resize all images to this size.
    
    Returns:
    - images (list): List of image arrays.
    - labels (list): Corresponding class labels.
    """
    images = []
    labels = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        print("class folder = ",class_folder)
        if class_folder == ".DS_Store":
            try:
                os.remove(class_folder)  # Delete the .DS_Store file
                print(f"Removed: {class_folder}")
            except Exception as e:
                print(f"Error removing {class_folder}: {e}")
            continue
        else:
            if os.path.isdir(class_path):  # Check if it is a folder
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(class_path, img_file)
                        try:
                            img = Image.open(img_path).convert('RGB')  # Ensure RGB mode
                            img = img.resize(image_size)  # Resize image
                            images.append(np.array(img))
                            labels.append(class_folder)  # Use folder name as label
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
            
    return np.array(images), np.array(labels)

def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Extract HOG features for a list of images.
    
    Args:
    - images (numpy array): Array of images (e.g., RGB or grayscale).
    - pixels_per_cell (tuple): Size of the cell for HOG.
    - cells_per_block (tuple): Number of cells per block for HOG.
    - orientations (int): Number of orientation bins for HOG.
    
    Returns:
    - features (numpy array): Array of HOG feature vectors.
    """
    features = []
    for img in images:
        # If the image is in RGB, convert it to grayscale
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)  # Simple grayscale conversion
        # Extract HOG features
        hog_features = hog(img, 
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           orientations=orientations,
                           block_norm='L2-Hys',  # Normalize blocks
                           feature_vector=True)
        features.append(hog_features)
    return np.array(features)



# Paths to training and testing folders
train_folder = "./Images/train"
test_folder = "./Images/test"
predict_folder = "./Images/predict"

# Load images and labels
image_size = (64, 64)  # Resize images to 64x64
X_train_images, y_train_labels = load_images_from_folder(train_folder, image_size)
X_test_images, y_test_labels = load_images_from_folder(test_folder, image_size)
X_predict_images, y_predict_labels = load_images_from_folder(predict_folder, image_size)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.transform(y_test_labels)
y_predict_test = le.transform(y_predict_labels)

# Extract HOG features
X_train_hog = extract_hog_features(X_train_images)
X_test_hog = extract_hog_features(X_test_images)
X_pred_hog = extract_hog_features(X_predict_images)

# Train the Random Forest model
#model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_hog, y_train)

# Test the model
y_pred = model.predict(X_test_hog)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Test the predition
y_pred_test = model.predict(X_pred_hog)
print(y_pred_test)