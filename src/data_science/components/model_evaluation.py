import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import joblib
from data_science import logger
from data_science.entity.config_entity import ModelEvaluationConfig
import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from data_science.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Aakash326/DS_main.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Aakash326"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = ""  # Keep secret in .env
        
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        
        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)
            
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metrics_file_name), data=scores)
            
            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Alternative approach: Log model as artifact instead of using log_model
            try:
                # Try the simple approach first - just log the model file
                mlflow.log_artifact(self.config.model_path, "model")
                logger.info("Model logged as artifact successfully.")
                
                # Optionally, save and log additional model info
                model_info = {
                    "model_type": str(type(model).__name__),
                    "feature_names": list(test_x.columns) if hasattr(test_x, 'columns') else None,
                    "target_column": self.config.target_column,
                    "model_path": str(self.config.model_path)
                }
                
                # Save model info as JSON and log it
                model_info_path = Path("model_info.json")
                save_json(path=model_info_path, data=model_info)
                mlflow.log_artifact(str(model_info_path), "model")
                
                # Clean up temporary file
                if model_info_path.exists():
                    model_info_path.unlink()
                    
            except Exception as e:
                logger.error(f"Failed to log model as artifact: {str(e)}")
                raise e
            
            logger.info("Model evaluation and logging completed successfully.")

