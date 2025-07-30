from data_science import logger
from data_science.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from data_science.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from data_science.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from data_science.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from typing import Type

def run_stage(stage_name: str, pipeline_class: Type):
    """Helper function to run a pipeline stage."""
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline = pipeline_class()
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == '__main__':
    run_stage("Data Ingestion stage", DataIngestionTrainingPipeline)
    run_stage("Data Validation stage", DataValidationTrainingPipeline)
    run_stage("Data Transformation stage", DataTransformationTrainingPipeline)
    run_stage("Model Trainer stage", ModelTrainerTrainingPipeline)