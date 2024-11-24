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
# import pandas as pd

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

# Machine Learning Sklearn Imports
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage import io, color, transform
from skimage.feature import hog
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load



# define some things in API
async def custom_lifespan(app: FastAPI):
    # Motor API allows us to directly interact with a hosted MongoDB server
    # In this example, we assume that there is a single client 
    # First let's get access to the Mongo client that allows interactions locally 
    app.mongo_client = motor.motor_asyncio.AsyncIOMotorClient()

    # new we need to create a database and a collection. These will create the db and the 
    # collection if they haven't been created yet. They are stored upon the first insert. 
    db = app.mongo_client.turidatabase
    app.collection = db.get_collection("labeledinstances")

    app.clf = {} # start app with no classifier
    app.sk_clf = {}

    yield 

    # anything after the yield can be used for clean up

    app.mongo_client.close()


# Create the FastAPI app
app = FastAPI(
    title="Machine Learning as a Service",
    summary="An application using FastAPI to add a ReST API to a MongoDB for data and labels collection.",
    lifespan=custom_lifespan,
)



# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.

# Annotated in python allows you to declare the type of a reference 
# and provide additional information related to it.
#   below we are declaring a "string" type with the annotation from BeforeValidator for a string type
#   this is the expectec setup for the pydantic Field below
# The validator is a pydantic check using the @validation decorator
# It specifies that it should be a strong before going into the validator
# we are not really using any advanced functionality, though, so its just boiler plate syntax
PyObjectId = Annotated[str, BeforeValidator(str)]


#========================================
#   Data store objects from pydantic 
#----------------------------------------
# These allow us to create a schema for our database and access it easily with FastAPI
# That might seem odd for a document DB, but its not! Mongo works faster when objects
# have a similar schema. 

'''
Create the data model and use strong typing. This also helps with the use of intellisense.

'''

class LabeledDataPoint(BaseModel):
    """
    Container for a single labeled data point.
    """

    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    feature: List[float] = Field(...) # feature data as array
    label: str = Field(...) # label for this data
    dsid: int = Field(..., le=50) # dataset id, for tracking different sets
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={ # provide an example for FastAPI to show users
            "example": {
                "feature": [-0.6,4.1,5.0,6.0],
                "label": "Walking",
                "dsid": 2,
            }
        },
    )


class LabeledDataPointCollection(BaseModel):
    """
    A container holding a list of instances.

    This exists because providing a top-level array in a JSON response can be a [vulnerability](https://haacked.com/archive/2009/06/25/json-hijacking.aspx/)
    """

    datapoints: List[LabeledDataPoint]


class FeatureDataPoint(BaseModel):
    """
    Container for a single labeled data point.
    """

    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    feature: List[float] = Field(...) # feature data as array
    dsid: int = Field(..., le=50) # dataset id, for tracking different sets
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={ # provide an example for FastAPI to show users
            "example": {
                "feature": [-0.6,4.1,5.0,6.0],
                "dsid": 2,
            }
        },
    )



#===========================================
#   FastAPI methods, for interacting with db 
#-------------------------------------------
# These allow us to interact with the REST server. All interactions with mongo should be 
# async, allowing the API to remain responsive even when servicing longer queries. 


@app.post(
    "/labeled_data/",
    response_description="Add new labeled datapoint",
    response_model=LabeledDataPoint,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_datapoint(datapoint: LabeledDataPoint = Body(...)):
    """
    Insert a new data point. Let user know the range of values inserted

    A unique `id` will be created and provided in the response.
    """
    
    # insert this datapoint into the database
    new_label = await app.collection.insert_one(
        datapoint.model_dump(by_alias=True, exclude=["id"])
    )

    # send back info about the record
    created_label = await app.collection.find_one(
        {"_id": new_label.inserted_id}
    )
    # also min/max of array, rather than the entire to array to save some bandwidth
    # the datapoint variable is a pydantic model, so we can access with properties
    # but the output of mongo is a dictionary, so we need to subscript the entry
    created_label["feature"] = [min(datapoint.feature), max(datapoint.feature)]

    return created_label


@app.get(
    "/labeled_data/{dsid}",
    response_description="List all labeled data in a given dsid",
    response_model=LabeledDataPointCollection,
    response_model_by_alias=False,
)
async def list_datapoints(dsid: int):
    """
    List all of the data for a given dsid in the database.

    The response is unpaginated and limited to 1000 results.
    """
    return LabeledDataPointCollection(datapoints=await app.collection.find({"dsid": dsid}).to_list(1000))


@app.get(
    "/max_dsid/",
    response_description="Get current maximum dsid in data",
    response_model_by_alias=False,
)
async def show_max_dsid():
    """
    Get the maximum dsid currently used 
    """

    if (
        datapoint := await app.collection.find_one(sort=[("dsid", -1)])
    ) is not None:
        return {"dsid":datapoint["dsid"]}

    raise HTTPException(status_code=404, detail=f"No datasets currently created.")



@app.delete("/labeled_data/{dsid}", 
    response_description="Delete an entire dsid of datapoints.")
async def delete_dataset(dsid: int):
    """
    Remove an entire dsid from the database.
    REMOVE AN ENTIRE DSID FROM THE DATABASE, USE WITH CAUTION.
    """

    # replace any underscores with spaces (to help support others)

    delete_result = await app.collection.delete_many({"dsid": dsid})

    if delete_result.deleted_count > 0:
        return {"num_deleted_results":delete_result.deleted_count}

    raise HTTPException(status_code=404, detail=f"DSID {dsid} not found")


#===========================================
#   Machine Learning Model Common Functions
#-------------------------------------------

# Parameters
img_size = (128, 128)

# Augment the images to create more diverse training set
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


#==============================================
#   Machine Learning methods (Scikit-learn SVM)
#----------------------------------------------

@app.get(
    "/train_model_svc/{dsid}", # instead of {dsid} should this be numpy array?
    response_description="Train a machine learning model for Support Vector Classification",
    response_model_by_alias=False,
)

# async def train_model_svc(dsid: int):
async def train_model_svc(dsid: int): 
    """
    Train the machine learning model on images provided by the user in a support vector classifier

    """
    datapoints = await app.collection.find({"dsid": dsid}).to_list(length=None)

    if len(datapoints) < 2:
        raise HTTPException(status_code=404, detail=f"DSID {dsid} has {len(datapoints)} datapoints.") 


    # Will no longer be using the SFrame
    # data = tc.SFrame(data={"target":[datapoint["label"] for datapoint in datapoints], 
    #     "sequence":np.array([datapoint["feature"] for datapoint in datapoints])}
    # )

    train_labels = np.array([datapoint["label"] for datapoint in datapoints])
    train_images = np.array([datapoint["feature"] for datapoint in datapoints])

    augmented_images, augmented_labels = augment_images(train_images, train_labels)
    X_train, y_train = create_dataset_with_hog(augmented_images,augmented_labels)

    X_train_pca = apply_pca(X_train)

    # Train an SVC
    model_svc = SVC()
    model_svc.fit(X_train_pca, y_train)
    
    # save model for use later, if desired
    model_svc.save("../models/svc_model_dsid%d"%(dsid))

    app.clf[dsid] = model_svc

    return {"summary":f"{model_svc}"}


@app.post(
    "/predict_svc",
    response_description="Predict Label from Datapoint",
)
async def predict_datapoint_svc(datapoint: FeatureDataPoint = Body(...)):
    """
    Post a feature set and get the label back.
    """
    # Reshape the feature array for prediction
    feature_array = np.array(datapoint.feature).reshape((1, -1))

    # Check if the model exists for the given dsid
    if datapoint.dsid not in app.clf:
        print("Loading SVC Model From file for DSID:", datapoint.dsid)

        try:
            # Load the scikit-learn model
            model_path = f"../models/svc_model_dsid{datapoint.dsid}.joblib"
            model = joblib.load(model_path)

            # Store the loaded model in the dictionary
            app.clf[datapoint.dsid] = model
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model for DSID {datapoint.dsid} not found. Model has not been trained. Error: {str(e)}"
            )

    # Predict the label using the loaded model
    model = app.clf[datapoint.dsid]
    predicted_label = model.predict(feature_array)[0]

    return {"predicted_label": predicted_label}


#=============================================
#   Machine Learning methods (Scikit-learn RF)
#---------------------------------------------

@app.get(
    "/train_model_rf/{dsid}", # instead of {dsid} should this be numpy array?
    response_description="Train a machine learning model for the given dsid",
    response_model_by_alias=False,
)

async def train_model_rf(dsid: int):
    """
    Train the machine learning model on images provided by the user in a support vector classifier

    """

    datapoints = await app.collection.find({"dsid": dsid}).to_list(length=None)

    if len(datapoints) < 2:
        raise HTTPException(status_code=404, detail=f"DSID {dsid} has {len(datapoints)} datapoints.") 


    # Will no longer be using the SFrame
    # data = tc.SFrame(data={"target":[datapoint["label"] for datapoint in datapoints], 
    #     "sequence":np.array([datapoint["feature"] for datapoint in datapoints])}
    # )

    train_labels = np.array([datapoint["label"] for datapoint in datapoints])
    train_images = np.array([datapoint["feature"] for datapoint in datapoints])

    augmented_images, augmented_labels = augment_images(train_images, train_labels)
    X_train, y_train = create_dataset_with_hog(augmented_images,augmented_labels)

    X_train_pca = apply_pca(X_train)

    # Train a Random Forest Model
    model_rf = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)
    model_rf.fit(X_train_pca, y_train)
    
    # save model for use later, if desired
    model_rf.save("../models/rf_model_dsid%d"%(dsid))

    app.clf[dsid] = model_rf

    return {"summary":f"{model_rf}"}


@app.post(
    "/predict_rf/",
    response_description="Predict Label from Datapoint",
)
async def predict_rf(datapoint: FeatureDataPoint = Body(...)):
    # async def predict_datapoint_turi(datapoint: FeatureDataPoint = Body(...)):
    """
    Post a feature set and get the label back

    """
    # Reshape the feature array for prediction
    feature_array = np.array(datapoint.feature).reshape((1, -1))

    # Check if the model exists for the given dsid
    if datapoint.dsid not in app.clf:
        print("Loading RF Model From file for DSID:", datapoint.dsid)

        try:
            # Load the scikit-learn model
            model_path = f"../models/rf_model_dsid{datapoint.dsid}.joblib"
            model = joblib.load(model_path)

            # Store the loaded model in the dictionary
            app.clf[datapoint.dsid] = model
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model for DSID {datapoint.dsid} not found. Model has not been trained. Error: {str(e)}"
            )

    # Predict the label using the loaded model
    model = app.clf[datapoint.dsid]
    predicted_label = model.predict(feature_array)[0]

    return {"predicted_label": predicted_label}