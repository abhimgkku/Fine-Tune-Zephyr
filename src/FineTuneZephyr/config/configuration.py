from src.FineTuneZephyr.constants import *
from src.FineTuneZephyr.logging import logger
from src.FineTuneZephyr.utils.common import read_yaml,create_directories
from src.FineTuneZephyr.entity import (DataIngestionConfig,
                                       #DataValidationConfig,
                                       #DataTransformationConfig,
                                       #TraningArgumentConfig,
                                       #LoraCongif,
                                       #ModelTrainingConfig,
                                       #ModelPredictionConfig
                                       
                                       )


class ConfigurationManager:
    def __init__(self,config_filepath=CONFIG_FILE_PATH,param_filepath= PARAMS_FILE_PATH):
        self.config= read_yaml(config_filepath)
        self.param = read_yaml(param_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:    
        config = self.config.Data_Ingestion

        

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(

            root_dir= config.root_dir,
            local_file_path=config.local_file_path
        )

        return data_ingestion_config
    