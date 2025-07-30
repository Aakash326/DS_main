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
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "39159c4f7843836d5a9af3ea120be37e9367afda"  # Keep secret in .env
        
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

    def log_into_mlflow_advanced(self):
        """Enhanced method that works with DagsHub limitations"""
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Aakash326/DS_main.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Aakash326"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "39159c4f7843836d5a9af3ea120be37e9367afda"
        
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        
        with mlflow.start_run() as run:
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)
            
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metrics_file_name), data=scores)
            
            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log metrics file
            mlflow.log_artifact(str(self.config.metrics_file_name))
            
            # Method 1: Try basic model logging (most compatible)
            try:
                # Save predictions for analysis
                predictions_df = pd.DataFrame({
                    'actual': test_y.iloc[:, 0].values,
                    'predicted': predicted_qualities
                })
                predictions_path = "predictions.csv"
                predictions_df.to_csv(predictions_path, index=False)
                mlflow.log_artifact(predictions_path, "results")
                
                # Log the model file directly
                mlflow.log_artifact(self.config.model_path, "model")
                
                # Log model metadata
                model_metadata = {
                    "model_type": str(type(model).__name__),
                    "feature_names": list(test_x.columns),
                    "target_column": self.config.target_column,
                    "n_features": len(test_x.columns),
                    "model_params": getattr(model, 'get_params', lambda: {})(),
                    "run_id": run.info.run_id
                }
                
                metadata_path = "model_metadata.json"
                save_json(path=Path(metadata_path), data=model_metadata)
                mlflow.log_artifact(metadata_path, "model")
                
                # Clean up temporary files
                import os
                for temp_file in [predictions_path, metadata_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                logger.info("Model evaluation and logging completed successfully using artifact logging.")
                
            except Exception as e:
                logger.error(f"Failed to log artifacts: {str(e)}")
                raise e

    def log_into_mlflow_local_only(self):
        """Method for local MLflow tracking only"""
        # Comment out DagsHub tracking for local testing
        # mlflow.set_tracking_uri("file:./mlruns")  # Local tracking
        
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        input_example = test_x.head()
        
        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)
            
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metrics_file_name), data=scores)
            
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # This should work locally
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=self.config.registered_model_name
            )
            
            logger.info("Model evaluation and logging completed successfully (local).")