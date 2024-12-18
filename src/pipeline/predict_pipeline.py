import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_objects


class PredictPipeline:
    # Empty Constructor.
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'src/components/artifacts/preprocessor.pkl' # To Transform Data.
            
            # Load Model and Preprocessor from Artifacts.
            model = load_objects(model_path)
            preprocessor = load_objects(preprocessor_path)
            
            # Transfrom the data(features).
            y_input = preprocessor.transform(features)
            
            return model.predict(y_input)
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_df(self):
        try:
            # Create Data Dict.
            data_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score],
            }
            
            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e, sys) 
        
        