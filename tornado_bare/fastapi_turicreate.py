#!/usr/bin/python
'''
In this example, we will use FastAPI as a gateway into a MongoDB database. We will use a REST style 
interface that allows users to initiate GET, POST, PUT, and DELETE requests. These commands will 
also be used to control certain functionalities with machine learning, using the ReST server to
function as a machine learning as a service, MLaaS provider. 

Specifically, we are creating an app that can take in motion sampled data and labels for 
segments of the motion data

The swift code for interacting with the interface is also available through the SMU MSLC class 
repository. 
Look for the https://github.com/SMU-MSLC/SwiftHTTPExample with branches marked for FastAPI and
turi create

To run this example in localhost mode only use the command:
fastapi dev fastapi_turicreate.py

Otherwise, to run the app in deployment mode (allowing for external connections), use:
fastapi run fastapi_turicreate.py

External connections will use your public facing IP, which you can find from the inet. 
A useful command to find the right public facing ip is:
ifconfig |grep "inet "
which will return the ip for various network interfaces from your card. If you get something like this:
inet 10.9.181.129 netmask 0xffffc000 broadcast 10.9.191.255 
then your app needs to connect to the netmask (the first ip), 10.9.181.129
'''

# For this to run properly, MongoDB should be running
#    To start mongo use this: brew services start mongodb-community@6.0
#    To stop it use this: brew services stop mongodb-community@6.0

# This App uses a combination of FastAPI and Motor (combining tornado/mongodb) which have documentation here:
# FastAPI:  https://fastapi.tiangolo.com 
# Motor:    https://motor.readthedocs.io/en/stable/api-tornado/index.html

# Maybe the most useful SO answer for FastAPI parallelism:
# https://stackoverflow.com/questions/71516140/fastapi-runs-api-calls-in-serial-instead-of-parallel-fashion/71517830#71517830
# Chris knows what's up 



import os
from typing import Optional, List
from enum import Enum
import numpy as np

# FastAPI imports
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response
from pydantic import ConfigDict, BaseModel, Field, EmailStr
from pydantic.functional_validators import BeforeValidator

from typing_extensions import Annotated

# Motor imports
from bson import ObjectId
import motor.motor_asyncio
from pymongo import ReturnDocument

# Machine Learning, Turi and Sklearn Imports
import turicreate as tc
from sklearn.neighbors import KNeighborsClassifier

from joblib import dump, load
import pickle
import numpy as np
import matplotlib.pyplot as plt


# # define some things in API
# async def custom_lifespan(app: FastAPI):
#     # Motor API allows us to directly interact with a hosted MongoDB server
#     # In this example, we assume that there is a single client 
#     # First let's get access to the Mongo client that allows interactions locally 
#     app.mongo_client = motor.motor_asyncio.AsyncIOMotorClient()

#     # new we need to create a database and a collection. These will create the db and the 
#     # collection if they haven't been created yet. They are stored upon the first insert. 
#     db = app.mongo_client.turidatabase
#     app.collection = db.get_collection("labeledinstances")

#     app.clf = {} # start app with no classifier
#     app.sk_clf = {}

#     yield 

#     # anything after the yield can be used for clean up

#     app.mongo_client.close()


# # Create the FastAPI app
# app = FastAPI(
#     title="Machine Learning as a Service",
#     summary="An application using FastAPI to add a ReST API to a MongoDB for data and labels collection.",
#     lifespan=custom_lifespan,
# )



# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.

# Annotated in python allows you to declare the type of a reference 
# and provide additional information related to it.
#   below we are declaring a "string" type with the annotation from BeforeValidator for a string type
#   this is the expectec setup for the pydantic Field below
# The validator is a pydantic check using the @validation decorator
# It specifies that it should be a strong before going into the validator
# we are not really using any advanced functionality, though, so its just boiler plate syntax
# PyObjectId = Annotated[str, BeforeValidator(str)]


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




# def load_images_from_folder(folder_path, image_size=(64, 64)):
#     """
#     Load images from folder and return as a list of numpy arrays with labels.
    
#     Args:
#     - folder_path (str): Path to the data folder (e.g., Train or Test).
#     - image_size (tuple): Resize all images to this size.
    
#     Returns:
#     - images (list): List of image arrays.
#     - labels (list): Corresponding class labels.
#     """
#     images = []
#     labels = []
#     for class_folder in os.listdir(folder_path):
#         class_path = os.path.join(folder_path, class_folder)
#         print("class folder = ",class_folder)
#         if class_folder == ".DS_Store":
#             try:
#                 os.remove(class_folder)  # Delete the .DS_Store file
#                 print(f"Removed: {class_folder}")
#             except Exception as e:
#                 print(f"Error removing {class_folder}: {e}")
#             continue
#         else:
#             if os.path.isdir(class_path):  # Check if it is a folder
#                 for img_file in os.listdir(class_path):
#                     if img_file.endswith('.jpg'):
#                         img_path = os.path.join(class_path, img_file)
#                         try:
#                             img = Image.open(img_path).convert('RGB')  # Ensure RGB mode
#                             img = img.resize(image_size)  # Resize image
#                             images.append(np.array(img))
#                             labels.append(class_folder)  # Use folder name as label
#                         except Exception as e:
#                             print(f"Error loading image {img_path}: {e}")
            
#     return np.array(images), np.array(labels)

# def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
#     """
#     Extract HOG features for a list of images.
    
#     Args:
#     - images (numpy array): Array of images (e.g., RGB or grayscale).
#     - pixels_per_cell (tuple): Size of the cell for HOG.
#     - cells_per_block (tuple): Number of cells per block for HOG.
#     - orientations (int): Number of orientation bins for HOG.
    
#     Returns:
#     - features (numpy array): Array of HOG feature vectors.
#     """
#     features = []
#     for img in images:
#         # If the image is in RGB, convert it to grayscale
#         if len(img.shape) == 3:
#             img = np.mean(img, axis=2)  # Simple grayscale conversion
#         # Extract HOG features
#         hog_features = hog(img, 
#                            pixels_per_cell=pixels_per_cell,
#                            cells_per_block=cells_per_block,
#                            orientations=orientations,
#                            block_norm='L2-Hys',  # Normalize blocks
#                            feature_vector=True)
#         features.append(hog_features)
#     return np.array(features)



# Paths to training and testing folders
# train_folder = "./Images/train"
# test_folder = "./Images/test"
# predict_folder = "./Images/predict"

# Load images and labels
# image_size = (64, 64)  # Resize images to 64x64
# X_train_images, y_train_labels = load_images_from_folder(train_folder, image_size)
# X_test_images, y_test_labels = load_images_from_folder(test_folder, image_size)
# X_predict_images, y_predict_labels = load_images_from_folder(predict_folder, image_size)

# Encode labels
# le = LabelEncoder()
# y_train = le.fit_transform(y_train_labels)
# y_test = le.transform(y_test_labels)
# y_predict_test = le.transform(y_predict_labels)

# Extract HOG features
# X_train_hog = extract_hog_features(X_train_images)
# X_test_hog = extract_hog_features(X_test_images)
# X_pred_hog = extract_hog_features(X_predict_images)

# Train the Random Forest model
#model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
# model = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)
# model.fit(X_train_hog, y_train)

# Test the model
y_pred = model.predict(X_test_hog)


# Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {accuracy:.2f}")

# print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=le.classes_))

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))


# Test the predition
y_pred_test = model.predict(X_pred_hog)
print(y_pred_test)

#========================================
#   Practice image import
#----------------------------------------
'''
We need to set things up so that we expect to receive a numpy array for each image
provided by the user from their phone. Right now we're just going to try this with 
local pictures to make sure we're importing and feeding the classifier correctly.
'''


'''
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import io, color, transform
from skimage.feature import hog
import os

# Create Training/Test image sets - Collect images from directories
train_images = []
test_images = []

train_directories = ['./Images/Dog', './Images/Not Dog'] # images from kaggle
test_directories = ['./Images/Test']

for dir in train_directories:
    for img in os.listdir(dir): 
        if img.endswith('.jpg'):
            image = io.imread(os.path.join(dir, img))
            train_images.append(image)
        else:
            exit


#positive_label = 'dog' # whatever user tells us it is
#negative_label = 'other' # not dog, basically
positive_label = 1 # whatever user tells us it is
negative_label = 0 # not dog, basically
#labels = [positive_label] * 4 + [negative_label] * 4
train_labels = [positive_label,positive_label,positive_label,positive_label,positive_label,positive_label,
                positive_label,positive_label,positive_label,positive_label,positive_label,positive_label,
                negative_label,negative_label,negative_label,negative_label,negative_label,negative_label,
                negative_label,negative_label,negative_label,negative_label,negative_label,negative_label]

# Create Test images set - Collect images from Test directory
for dir in test_directories:
    for img in os.listdir(dir): 
        if img.endswith('.jpg'):
            test_image = io.imread(os.path.join(dir, img))
            test_images.append(test_image)
        else:
            exit

def create_dataset(img_list, label_list):

    # Handle the image data and prepare for training
    #h, w = 224, 224
    h, w = 128, 128

    g_img = [color.rgb2gray(x) for x in img_list] # convert to grayscale
    resize_img = [transform.resize(img, (h,w), anti_aliasing=True) for img in g_img] # resize

    # SVC and RF expects a flat numpy array
    #X = np.array([img.flatten() for img in resize_img])

    # Extract HOG (Histogram of Oriented Gradients - from ChatGPT (CITE REF) )
    hog_img = [hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm='L2-Hys')
               for img in resize_img]
    
    X = np.array(hog_img)

    # Handle the label data and get it ready to train  
    le = LabelEncoder()
    y = le.fit_transform(label_list) # dog: 0, not dog: 1
    return X, y

def train_svc(X_tr,y_lbl):
    # Train an SVC
    svc.fit(X_tr, y_lbl)

    print("SVC: Training complete!")

    # Training and Test Accuracy
    print("Training Accuracy:", svc.score(X_tr, y_lbl))
    return

def predict_svc(X_tst):
    print("Predict using SVC")
    y_pred = svc.predict(X_tst)
    print(y_pred)
    return

def train_rf(X_tr,y_lbl):
   # print("train_rf: type-X_tr",type(X_tr))
   # print("train_rf: len-X_tr",len(X_tr))
   # print([len(x) for x in X_tr])

    # Train a Random Forest
    rf.fit(X_tr, y_lbl)

    print("RF: Training complete!")

    # Training and Test Accuracy
    print("Training Accuracy:", rf.score(X_tr, y_lbl))
    return

def predict_rf(X_tst):

    print("Predict using RF")
    y_pred = rf.predict(X_tst)
    print(y_pred)
    return

# Create training dataset
#X, y = create_dataset(train_images[:8],train_labels)
X, y = create_dataset(train_images[:24],train_labels)

# Print data shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

 # Create some test examples
#X_test, _ = create_dataset(test_images[8:], []) # no labels for test data
X_test, _ = create_dataset(test_images, []) # no labels for test data
print("X_test shape:", X_test.shape)

#Instantiate SVC and RF classifiers
svc = SVC() # model selection 1
rf= RandomForestClassifier()  # model selection 2
#rf= RandomForestClassifier(  # model selection 2
#    n_estimators=100,
#    max_depth=10,
#    min_samples_split=5,
#    min_samples_leaf=2,
#    max_features='sqrt',
#    bootstrap=True,
#    random_state=42
#) 

selected_model = int(input("Input Model to use (1-SVC, 2-RF)")) 

# selected_model = 1 # select svc

if selected_model == 1 :
    print("Selected Model: SVC")
    train_svc(X,y)
    predict_svc(X_test)
else:
    print("Selected Model: RF")
    train_rf(X,y)
    predict_rf(X_test)

'''

# #========================================
# #   Data store objects from pydantic 
# #----------------------------------------
# # These allow us to create a schema for our database and access it easily with FastAPI
# # That might seem odd for a document DB, but its not! Mongo works faster when objects
# # have a similar schema. 

# '''Create the data model and use strong typing. This also helps with the use of intellisense.
# '''
# class LabeledDataPoint(BaseModel): # Need to update this function
#     """
#     Container for a single labeled data point.
#     """

#     # This will be aliased to `_id` when sent to MongoDB,
#     # but provided as `id` in the API requests and responses.
#     id: Optional[PyObjectId] = Field(alias="_id", default=None)
#     feature: List[float] = Field(...) # feature data as array
#     label: str = Field(...) # label for this data
#     dsid: int = Field(..., le=50) # dataset id, for tracking different sets
#     model_config = ConfigDict(
#         populate_by_name=True,
#         arbitrary_types_allowed=True,
#         json_schema_extra={ # provide an example for FastAPI to show users
#             "example": {
#                 "feature": [-0.6,4.1,5.0,6.0],
#                 "label": "Walking",
#                 "dsid": 2,
#             }
#         },
#     )


# class LabeledDataPointCollection(BaseModel):
#     """
#     A container holding a list of instances.

#     This exists because providing a top-level array in a JSON response can be a [vulnerability](https://haacked.com/archive/2009/06/25/json-hijacking.aspx/)
#     """

#     datapoints: List[LabeledDataPoint]


# class FeatureDataPoint(BaseModel): # Note: Going to have to update this function
#     """
#     Container for a single labeled data point.
#     """

#     # This will be aliased to `_id` when sent to MongoDB,
#     # but provided as `id` in the API requests and responses.
#     id: Optional[PyObjectId] = Field(alias="_id", default=None)
#     feature: List[float] = Field(...) # feature data as array
#     dsid: int = Field(..., le=50) # dataset id, for tracking different sets
#     model_config = ConfigDict(
#         populate_by_name=True,
#         arbitrary_types_allowed=True,
#         json_schema_extra={ # provide an example for FastAPI to show users
#             "example": {
#                 "feature": [-0.6,4.1,5.0,6.0],
#                 "dsid": 2,
#             }
#         },
#     )



# #===========================================
# #   FastAPI methods, for interacting with db 
# #-------------------------------------------
# # These allow us to interact with the REST server. All interactions with mongo should be 
# # async, allowing the API to remain responsive even when servicing longer queries. 


# @app.post(
#     "/labeled_data/",
#     response_description="Add new labeled datapoint",
#     response_model=LabeledDataPoint,
#     status_code=status.HTTP_201_CREATED,
#     response_model_by_alias=False,
# )
# async def create_datapoint(datapoint: LabeledDataPoint = Body(...)):
#     """
#     Insert a new data point. Let user know the range of values inserted

#     A unique `id` will be created and provided in the response.
#     """
    
#     # insert this datapoint into the database
#     new_label = await app.collection.insert_one(
#         datapoint.model_dump(by_alias=True, exclude=["id"])
#     )

#     # send back info about the record
#     created_label = await app.collection.find_one(
#         {"_id": new_label.inserted_id}
#     )
#     # also min/max of array, rather than the entire to array to save some bandwidth
#     # the datapoint variable is a pydantic model, so we can access with properties
#     # but the output of mongo is a dictionary, so we need to subscript the entry
#     created_label["feature"] = [min(datapoint.feature), max(datapoint.feature)]

#     return created_label


# @app.get(
#     "/labeled_data/{dsid}",
#     response_description="List all labeled data in a given dsid",
#     response_model=LabeledDataPointCollection,
#     response_model_by_alias=False,
# )
# async def list_datapoints(dsid: int):
#     """
#     List all of the data for a given dsid in the database.

#     The response is unpaginated and limited to 1000 results.
#     """
#     return LabeledDataPointCollection(datapoints=await app.collection.find({"dsid": dsid}).to_list(1000))


# @app.get(
#     "/max_dsid/",
#     response_description="Get current maximum dsid in data",
#     response_model_by_alias=False,
# )
# async def show_max_dsid():
#     """
#     Get the maximum dsid currently used 
#     """

#     if (
#         datapoint := await app.collection.find_one(sort=[("dsid", -1)])
#     ) is not None:
#         return {"dsid":datapoint["dsid"]}

#     raise HTTPException(status_code=404, detail=f"No datasets currently created.")



# @app.delete("/labeled_data/{dsid}", 
#     response_description="Delete an entire dsid of datapoints.")
# async def delete_dataset(dsid: int):
#     """
#     Remove an entire dsid from the database.
#     REMOVE AN ENTIRE DSID FROM THE DATABASE, USE WITH CAUTION.
#     """

#     # replace any underscores with spaces (to help support others)

#     delete_result = await app.collection.delete_many({"dsid": dsid})

#     if delete_result.deleted_count > 0:
#         return {"num_deleted_results":delete_result.deleted_count}

#     raise HTTPException(status_code=404, detail=f"DSID {dsid} not found")


#===========================================
#   Machine Learning Model Common Functions
#-------------------------------------------
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage import io, color, transform
from skimage.feature import hog
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import os

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
train_data = data[~data['Filename'].str.contains("Test")] # Filepaths not containing "Test"
test_data = data[data['Filename'].str.contains("Test")] # Everything else

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

        # # Add noise
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
    pca = PCA(n_components=explained_variance)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
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

    # If labels are provided, encode them

    le = LabelEncoder()
    y = le.fit_transform(label_list)  # dog: 0, not dog: 1

    return X, y


#===========================================
#   Machine Learning methods (Scikit-learn SVM)
#-------------------------------------------

@app.get(
    "/train_model_svc/{dsid}", # instead of {dsid} should this be numpy array?
    response_description="Train a machine learning model for Support Vector Classification",
    response_model_by_alias=False,
)

# async def train_model_svc(dsid: int):
async def train_model_svc(train_images: list, train_labels: list):
    """
    Train the machine learning model on images provided by the user in a support vector classifier

    """

    augmented_images, augmented_labels = augment_images(train_images, train_labels)
    X_train, y_train = create_dataset_with_hog(augmented_images,augmented_labels)

    X_train_pca = apply_pca(X_train)

    # Train an SVC
    model_svc = SVC()
    model_svc.fit(X_train_pca, y_train)
    
    # save model for use later, if desired
    model_svc.save("../models/svc_model")

    ### Needs updated ###
    app.clf[dsid] = model_svc

    return {"summary":f"{model_svc}"}


@app.post(
    "/predict_svc/",
    response_description="Predict Label from Datapoint",
)
async def predict_svc(test_image: np.array):
    """
    Post a feature set and get the label back

    """
    # X_test needs to be FeatureDataPoint, or something specific
    # Expect np.array from user photo
    X_test = # something with test_image parameter
    X_test_pca = apply_pca(X_test)
    y_pred = model_svc.predict(X_test_pca) 
    print("Predictions:", y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # place inside an SFrame (that has one row)
    data = tc.SFrame(data={"sequence":np.array(datapoint.feature).reshape((1,-1))})

    # # Flipped Module 4 Update: Section 3 step 12 part 1 and 2
    # # Updated to check if dsid exists, handle errors
    # # if(app.clf == {}):
    # if datapoint.dsid not in app.clf: # new if statement
    #     print("Loading Turi Model From file for DSID: ", datapoint.dsid)

    #     try:
    #         # Try to set model equal to the correct model from the dictionary
    #         model = tc.load_model("../models/turi_model_dsid%d"%(datapoint.dsid))
    #         # Updated to use dictionary
    #         app.clf[datapoint.dsid] = model
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code = 404,
    #             detail=f"Model for DSID {datapoint.dsid} not found. Model has not been trained. Error {str(e)}"
    #         )

        # TODO: what happens if the user asks for a model that was never trained?
        #       or if the user asks for a dsid without any data? 
        #       need a graceful failure for the client...

    # Updated to use dictionary
    pred_label = app.clf[datapoint.dsid].predict(data)
    return {"prediction":str(pred_label)}


#===========================================
#   Machine Learning methods (Scikit-learn RF)
#-------------------------------------------

@app.get(
    "/train_model_rf/{dsid}", # instead of {dsid} should this be numpy array?
    response_description="Train a machine learning model for the given dsid",
    response_model_by_alias=False,
)

async def train_model_rf(dsid: int):
    """
    Train the machine learning model on images provided by the user in a support vector classifier

    """

    augmented_images, augmented_labels = augment_images(train_images, train_labels)
    X_train, y_train = create_dataset_with_hog(augmented_images,augmented_labels)
    X_test, y_test = create_dataset_with_hog(test_images, test_labels)

    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    # Train a Random Forest Model
    # model = SVC()
    # model.fit(X_train_pca, y_train)
    model_rf = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)
    model_rf.fit(X_train_hog, y_train)
    # model = tc.classifier.create(data,target="target",verbose=0)# training
    
    # save model for use later, if desired
    model_rf.save("../models/rf_model_dsid%d"%(dsid)) # Will dsid play a role in our solution?

    # save this for use later 

    # Flipped Module 4 Update: Section 3 step 11 part 1
    # Update app.clf to be a dictionary
    app.clf[dsid] = model_rf

    return {"summary":f"{model_rf}"}


@app.post(
    "/predict_svc/",
    response_description="Predict Label from Datapoint",
)
async def predict_svc(datapoint: FeatureDataPoint = Body(...)):
    """
    Post a feature set and get the label back

    """
    # X_test needs to be FeatureDataPoint, or something specific
    # Expect np.array from user photo
    y_pred = model_rf.predict(X_test_hog)
    # y_pred = model_rf.predict(X_test_pca) 
    print("Predictions:", y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # place inside an SFrame (that has one row)
    data = tc.SFrame(data={"sequence":np.array(datapoint.feature).reshape((1,-1))})

    # # Flipped Module 4 Update: Section 3 step 12 part 1 and 2
    # # Updated to check if dsid exists, handle errors
    # # if(app.clf == {}):
    # if datapoint.dsid not in app.clf: # new if statement
    #     print("Loading Turi Model From file for DSID: ", datapoint.dsid)

    #     try:
    #         # Try to set model equal to the correct model from the dictionary
    #         model = tc.load_model("../models/turi_model_dsid%d"%(datapoint.dsid))
    #         # Updated to use dictionary
    #         app.clf[datapoint.dsid] = model
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code = 404,
    #             detail=f"Model for DSID {datapoint.dsid} not found. Model has not been trained. Error {str(e)}"
    #         )

        # TODO: what happens if the user asks for a model that was never trained?
        #       or if the user asks for a dsid without any data? 
        #       need a graceful failure for the client...

    # Updated to use dictionary
    pred_label = app.clf[datapoint.dsid].predict(data)
    return {"prediction":str(pred_label)}