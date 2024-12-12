# Data-Transformation is the process of converting data from one formate, structure or value to another "to make it suitable for analysis, storage and other operations."
# eg: Cleaning-data, Normalizing-Data, Data-reduction, Handling Missing-Values....

import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# To create data-trasformation pipeline.
from sklearn.compose import ColumnTransformer
# To handel missing-values.
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # To store Paths
    
    # Path to store pre-processed data for later use.
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl') 
    

class DataTransformation:
    def __init__(self):
        # Contains all the paths.
        self.data_transformation_config = DataTransformationConfig()
        
    def data_transformer(self):
        try:
            num_features = ['reading score', 'writing score']
            cat_features = [
                'gender',
                'race/ethnicity',
                'parental level of education',	
                'lunch',	
                'test preparation course'
            ]
            
            # Create PipeLines for numerical and categorical features.
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            logging.info('Pipeline created for both Numerical Feature Transformation.')
            
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
                ]
            )
 
            logging.info('Pipeline created for both Categorical Feature Transformation.')
            
            # Combine Numerical and categorical pipeline(actual trasformation).
            preprocessor = ColumnTransformer(
                [('num_pipeline', num_pipeline, num_features), ('cat_pipeline', cat_pipeline, cat_features)]
                # Syntax:
                # ('pipeline type', 'created-pipeline', 'features/columns')
            )
            
            logging.info('Data-Transformation completed.')
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read Train and Test Dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Loaded Train and Test Datasets.')
            
            logging.info('Obtaining preprocessor-object.......')
            preprocessor_obj = self.data_transformer()
            
            target = 'math score'
            num_features = ['reading score', 'writing score']
            
            # Separate input and output(target) for the model.
            train_input_df = train_df.drop(columns=[target], axis=1)
            train_target_df = train_df[target]
            
            test_input_df = test_df.drop(columns=[target], axis=1)
            test_target_df = test_df[target]
            
            # Transform separated data(only inputs), returns Numpy-Arrays.
            train_input = preprocessor_obj.fit_transform(train_input_df)
            test_input = preprocessor_obj.transform(test_input_df)
            
            logging.info('Data Transformation Completed.')
            
            # Concatinate Arrays(input and target) along the columns(second axis).
            train_arr = np.c_[train_input, np.array(train_target_df)]
            test_arr = np.c_[test_input, np.array(test_target_df)]

            logging.info('Data Concatinated Completed.')
            
            # Save object to the pickle-file.
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            logging.info('Saved Preprocessing Object.')
            
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
             
        except Exception as e:
            raise CustomException(e, sys)

