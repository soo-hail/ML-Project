import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

# NOTE: WE GOT REGRESSION PROBLEM, FINDING MATH_SCORE BASED ON DIFFERENT(MORE THAN ONE) FEATURES.

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    from sklearn.metrics import r2_score

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            # Split the Data
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])

            # Dict of Models.
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(verbose=False),
                'AdaBoost Regressor': AdaBoostRegressor(),
            }
            
            # Evaluate Models using the provided 'evaluate_models' function.
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # Get Best Model based on highest r2_score.
            best_model = max(model_report, key=model_report.get)
            
            # Check if the best model's r2 score is below the threshold.
            if model_report[best_model] < 0.6:
                raise CustomException(f'No Best Model found. Best Model ({best_model}) r² Score: {model_report[best_model]} is below threshold.')
            
            logging.info(f'Best Model Found: {best_model} with r² Score: {model_report[best_model]}')

            # Retrieve the trained Best Model from models
            best_model_trained = models[best_model]
            
            # Save the trained Best Model
            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model_trained
            )
            
            logging.info(f'Best Model ({best_model}) saved successfully.')
            
            # Predict on the test data using the trained best model.
            y_test_pred = best_model_trained.predict(X_test)
            
            # Optionally: Calculate r2_score on the test set for additional confirmation.
            test_r2 = r2_score(y_test, y_test_pred)
            logging.info(f'Best Model Test r² Score: {test_r2}')
            
            return test_r2

        except Exception as e:
            raise CustomException(e, sys)

