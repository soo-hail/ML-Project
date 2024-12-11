# CONTAINS ALL THE COMMON METHODS, FUNCTION THAT ARE USED IN ENTIRE PROJECTS.

import os
import sys
import dill
# 'dill' is a pickle-module, used to save python objects to a file.
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Traverse Models.
        for key in models.keys():
            model = models[key]

            # Train the model.
            model.fit(X_train, y_train)
            
            y_test_pred = model.predict(X_test)

            # Calculate r2_score
            model_r2score = r2_score(y_test, y_test_pred)
            
            # Store Test r² Score in Report.
            report[key] = model_r2score

            models[key] = model  # Update models dict with the trained model.

        return report
    
    except Exception as e:
        raise CustomException(e, sys)

