# Data-Ingestion is a process of "gathering data" from different sources and moving it into a central system.
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation 

# NOTE:
# dataclass is used to avoid writing boilerplate code, it automatically generates common methods like __init__, __repr__, and __eq__. 

@dataclass
class DataIngestionConfig:
    # To store data paths.
    raw_data_path: str = os.path.join('artfacts', 'data.csv')
    train_data_path: str = os.path.join('artfacts', 'train.csv')
    test_data_path: str = os.path.join('artfacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Contains all the paths.
        
    def initiate_data_ingestion(self):
        # To read the data from the centralized system.
        logging.info('Entered into initiate_data_ingestion')

        try:
            # Load the dataset into a DataFrame
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Dataset loaded into DataFrame')

            # Create the directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved.')

            # Split data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training data
            train_set.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            logging.info('Training data saved.')

            # Save test data
            test_set.to_csv(self.ingestion_config.test_data_path, header=True, index=False)
            logging.info('Test data saved.')

            return (
                # Return paths for data transformation
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    # Create an object of DataIngestion
    obj = DataIngestion()

    # Call the data ingestion method
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    # Data Transformation.
    data_tranformation = DataTransformation()
    data_tranformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
