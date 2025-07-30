from data_science.config.configuration import ConfigurationManager
from data_science.components.model_evaluation import ModelEvaluation
from data_science import logger
from pathlib import Path

STAGE_NAME = "Model Evaluation stage"
class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow() # This method handles the full evaluation and logging process
    
