from src.FineTuneZephyr.config.configuration import ConfigurationManager
from src.FineTuneZephyr.components.data_validation import DataValidation


class DataValidationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config_file= ConfigurationManager()
        data_validation_config= config_file.get_data_valdation_config()
        Data_Valdation= DataValidation(data_validation_config)
        Data_Valdation.Validation_Data()
       