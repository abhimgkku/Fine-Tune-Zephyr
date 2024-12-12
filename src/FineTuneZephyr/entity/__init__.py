from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    local_file_path:Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path   
    local_file_validation: Path 


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path    
    local_data_file: Path



@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: str
    traning_data_file: Path   
    #model_check_point: str 
    model_name: str
    tokenizer_name: str


@dataclass(frozen=True)
class TraningArgumentConfig:
    bits: int
    disable_exllama: bool
    model_id: str
    device_map: str
    use_cache: bool
    output_dir: str
    batch_size: int
    grad_accumulation_steps: int
    optimizer: str
    lr: float
    lr_scheduler: str
    save_strategy: str
    logging_steps: int
    num_train_epoch: int
    max_steps: int
    fp16: bool
    push_to_hub: bool


@dataclass(frozen=True)
class LoraCongif:
    lora_r: int
    lora_alpha: int 
    lora_dropout: float
    bias: str
    task_type: str
    target_modules: List

@dataclass(frozen=True)
class ModelPredictionConfig:
    model_name: Path
    tokenizer_name: Path    