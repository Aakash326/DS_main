import os
import urllib.request as request
import zipfile
from data_science import logger
from data_science.entity.config_entity import DataTransformationConfig
import pandas as pd
from sklearn.model_selection import train_test_split


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data= pd.read_csv(self.config.data_path)
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

        train_set.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test_set.to_csv(os.path.join(self.config.root_dir, "test.csv"), index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train_set.shape)
        logger.info(test_set.shape)

        print(train_set.shape)
        print(test_set.shape)
        