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
            input_example = test_x.head()

            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities = model.predict(test_x)
                rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(path=Path(self.config.metrics_file_name), data=scores)

                mlflow.log_params(self.config.all_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example,
                    registered_model_name=self.config.registered_model_name,
                
        )

                logger.info("Model evaluation and logging completed successfully.")