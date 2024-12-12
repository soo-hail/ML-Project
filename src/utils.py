# CONTAINS ALL THE COMMON METHODS, FUNCTION THAT ARE USED IN ENTIRE PROJECTS.

import os
import sys
import dill
# 'dill' is a pickle-module, used to save python objects to a file.
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
# for GridSearch.
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# Function to save objects in pickle-file.
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Open the file in 'write-binary' mode.
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
            
    except Exception as e:
        raise CustomException(e, sys)
    
# Function to evaluate Modles Based on accuracy(r2_score) to choose one-Model(best).
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        # Traverse Models.
        for key in models.keys():
            model = models[key]

            # Retrieve hyperparameters for the model.
            para = params.get(key, {}) 
            
            # Grid Search.
            gs = GridSearchCV(model, para, cv=3, scoring='r2', n_jobs=-1)
            
            # Train the model using GridSearchCV.
            gs.fit(X_train, y_train)
            
            # Get the best model with optimal parameters.
            best_model = gs.best_estimator_
            
            # Predict on the test data using the best model.
            y_test_pred = best_model.predict(X_test)
            
            # Calculate r² score for the test set.
            model_r2score = r2_score(y_test, y_test_pred)
            
            report[key] = model_r2score

            # Update the models dictionary with the best model.
            models[key] = best_model

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
# Function to Load Objects from Artifacts.
def load_objects(file_path):
    try:
        with open(file_path, 'rb') as f:
            return dill.load(f)
        
    except Exception as e:
        raise CustomException(e, sys)
