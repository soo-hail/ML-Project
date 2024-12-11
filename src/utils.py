# CONTAINS ALL THE COMMON METHODS, FUNCTION THAT ARE USED IN ENTIRE PROJECTS.

import os
import sys
import dill
# 'dill' is a pickle-module, used to save python objects to a file.
import numpy as np
import pandas as pd

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