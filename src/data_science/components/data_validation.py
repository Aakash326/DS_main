import pandas as pd
from data_science import logger
from data_science.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True
            data = pd.read_csv(self.config.data_file, sep=self.config.delimiter)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema

            # Check if column sets are identical
            if set(all_cols) != set(all_schema.keys()):
                validation_status = False
                logger.warning(f"Column mismatch. Data columns: {set(all_cols)}, Schema columns: {set(all_schema.keys())}")
            else:
                # If columns match, check data types
                for col, dtype in all_schema.items():
                    if str(data[col].dtype) != dtype:
                        validation_status = False
                        logger.warning(f"Data type mismatch for column '{col}'. Expected: {dtype}, Found: {data[col].dtype}")
                        break # Exit on first mismatch

            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Data validation status: {validation_status}")
            return validation_status

        except Exception as e:
            logger.exception(e)
            # Write failure status if an exception occurs
            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: False (Error: {e})")
            raise e