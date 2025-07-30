from data_science.config.configuration import ConfigurationManager
from data_science.components.data_transformation import DataTransformation
from data_science import logger
from pathlib import Path

STAGE_NAME = "Data Transformation stage"
class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()

            with open(data_validation_config.status_file, "r") as file:
                status = file.read().split(" ")[-1]
            
            if status == "True":
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_spliting()
            else:
                raise Exception("Data Validation failed. Cannot proceed with Data Transformation.")
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
