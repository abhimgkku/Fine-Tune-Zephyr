from src.FineTuneZephyr.config.configuration import ConfigurationManager
from src.FineTuneZephyr.components.data_ingestion import DataIngestion





class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config_file= ConfigurationManager()
        data_Ingestion_config= config_file.get_data_ingestion_config()
        Data_Ingestion= DataIngestion(data_Ingestion_config)
        Data_Ingestion.Download_Data()