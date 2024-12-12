import os
from datasets import Dataset
from src.FineTuneZephyr.entity import DataTransformationConfig
from src.FineTuneZephyr.logging import logger
from pathlib import Path



class Datatransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig) -> None:
        self.data_transformation_config = data_transformation_config


    def process_data_sample(self,example):
        processed_example = "<|system|>\n You are a support chatbot who helps with user queries chatbot who always responds in the style of a professional.\n<|user|>\n" + example["instruction"] + "\n<|assistant|>\n" + example["response"]
        return processed_example
    
    def create_dataset(self,data):
        df = data.to_pandas()
        df["text"] = df[["instruction", "response"]].apply(lambda x: self.process_data_sample(x), axis=1)
        processed_data = Dataset.from_pandas(df[["text"]])
        return processed_data



    def save_transformed_data(self):
         
         logger.info(f"Load Dataset from {self.data_transformation_config.local_data_file}")
         dataset = Dataset.from_csv(self.data_transformation_config.local_data_file)
         logger.info('transform dataset as Zephyr model required')
         transformed_datset = self.create_dataset(dataset)
         
         transformed_datset.save_to_disk(self.data_transformation_config.root_dir)
         logger.info(f"saved transformed dataset to {self.data_transformation_config.root_dir}")