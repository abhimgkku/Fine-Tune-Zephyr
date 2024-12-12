from src.FineTuneZephyr.config.configuration import ConfigurationManager
from src.FineTuneZephyr.components.data_transformation import Datatransformation





class DataTransformationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config_file= ConfigurationManager()
        data_transform_config= config_file.get_data_transfomation_config()
        Data_Transform= Datatransformation(data_transform_config)
        Data_Transform.save_transformed_data()