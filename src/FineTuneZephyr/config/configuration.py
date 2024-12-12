from src.FineTuneZephyr.constants import *
from src.FineTuneZephyr.logging import logger
from src.FineTuneZephyr.utils.common import read_yaml,create_directories
from src.FineTuneZephyr.entity import (DataIngestionConfig,
                                       DataValidationConfig,
                                       DataTransformationConfig,
                                       TraningArgumentConfig,
                                       LoraCongif,
                                       ModelTrainingConfig,
                                       ModelPredictionConfig
                                       
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
    
    def get_data_transfomation_config(self)->DataTransformationConfig:
        config= self.config.Data_Transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(

            root_dir=config.root_dir,
            local_data_file = config.local_data_file
        )

        return data_transformation_config
    
    def get_data_valdation_config(self)-> DataValidationConfig:
        config = self.config.Data_Validaton

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(

            root_dir=config.root_dir,
            status_file=config.status_file,
            local_file_validation = config.local_file_validation
        )

        return data_validation_config

    
    def get_model_training_config(self)-> ModelTrainingConfig:
        config = self.config.Model_training

        create_directories([config.root_dir])

        model_training_config= ModelTrainingConfig(

            root_dir = config.root_dir,
            traning_data_file =  config.traning_data_file,
            model_name = config.model_name,
            tokenizer_name = config.tokenizer_name


        )

        return model_training_config
    

    def get_trainingargumentconfig(self)->TraningArgumentConfig:
        param = self.param.TrainingArguments

        train_argument = TraningArgumentConfig(
                bits = param.bits,
                disable_exllama = param.disable_exllama,
                model_id = param.model_id,
                device_map = param.device_map,
                use_cache = param.use_cache,
                output_dir = param.output_dir,
                batch_size = param.batch_size,
                grad_accumulation_steps = param.grad_accumulation_steps,
                optimizer = param.optimizer,
                lr = param.lr,
                lr_scheduler = param.lr_scheduler,
                save_strategy = param.save_strategy,
                logging_steps = param.logging_steps,
                num_train_epoch = param.num_train_epoch,
                max_steps = param.max_steps,
                fp16 = param.fp16,
                push_to_hub = param.push_to_hub,
                max_seq_length = param.max_seq_length,
                packing = param.packing,
             

        )

        return train_argument
    

    def get_loraconfiguration(self)-> LoraCongif:
        param = self.param.LoraConfiguration

        lora_config = LoraCongif(
                    lora_r = param.lora_r,
                    lora_alpha = param.lora_alpha,
                    lora_dropout = param.lora_dropout,
                    bias =  param.bias,
                    task_type = param.task_type,
                    target_modules = param.target_modules,
            


        )

        return lora_config
    


    def get_prediction_config(self)-> ModelPredictionConfig:
        config= self.config.Model_Prediction

        prediction_config = ModelPredictionConfig(

            model_name= config.model_name,
            tokenizer_name= config.tokenizer_name
        )

        return prediction_config