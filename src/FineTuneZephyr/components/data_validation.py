import os
from pathlib import Path
from src.FineTuneZephyr.logging import logger
from src.FineTuneZephyr.entity import DataValidationConfig
from src.FineTuneZephyr.utils.common import WriteFile





class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig) -> None:
        self.data_validation_config = data_validation_config


    def Validation_Data(self):


        if os.path.exists(self.data_validation_config.local_file_validation):
            WriteFile(self.data_validation_config.status_file,"True")
                
            logger.info(f"Data file is available for data processing in {self.data_validation_config.local_file_validation}")

        else:
            WriteFile(self.data_validation_config.status_file,"False")    
            logger.info(f"Data file is not available for data processing in {self.data_validation_config.local_file_validation}")