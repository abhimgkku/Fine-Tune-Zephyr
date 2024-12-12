import os
from src.FineTuneZephyr.config.configuration import ModelTrainingConfig,TraningArgumentConfig,LoraCongif
from peft import LoraConfig, PeftModel,get_peft_model
from peft import prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments)

class ModelTraining:
    def __init__(self,model_training_config:ModelTrainingConfig, training_parameters:TraningArgumentConfig, lora_config:LoraCongif) -> None:
        self.model_training_config = model_training_config
        self.training_parameters = training_parameters
        self.lora_config = lora_config
    
    # Load Base Model
    def Base_model(self):
        bnb_config = GPTQConfig(
                    bits=self.training_parameters.bits,
                    disable_exllama=self.training_parameters.disable_exllama,
                    tokenizer=self.Zephyr_tokenizer()
                                )

        model = AutoModelForCausalLM.from_pretrained(
                self.training_parameters.model_id,
                quantization_config=bnb_config,
                device_map=self.training_parameters.device_map
                                                    )

        print("\n====================================================================\n")
        print("\t\t\tDOWNLOADED MODEL")
        print(model)
        print("\n====================================================================\n")

        model.config.use_cache=self.training_parameters.use_cache
        model.config.pretraining_tp=1
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        print("\n====================================================================\n")
        print("\t\t\tMODEL CONFIG UPDATED")
        print("\n====================================================================\n")

        return model
    
    # Load Zephyr tokenizer

    def Zephyr_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.training_parameters.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    

    # Load LoRA Configuration

    def peft_Lora_configuration(self):
        peft_config = LoraConfig(
                            r=self.lora_config.lora_r,
                            lora_alpha=self.lora_config.lora_alpha,
                            lora_dropout=self.lora_config.lora_dropout,
                            bias=self.lora_config.bias,
                            task_type=self.lora_config.task_type,
                            target_modules=self.lora_config.target_modules
                            )
        return peft_config
        
    def Load_dataset(self):

        dataset = load_from_disk(self.model_training_config.traning_data_file)
        #dataset = dataset.select(range(2))

        return dataset
    
    def Training_Argument(self):
        training_arguments = TrainingArguments(
        output_dir=self.model_training_config.model_name,
        per_device_train_batch_size=self.training_parameters.batch_size,
        gradient_accumulation_steps=self.training_parameters.grad_accumulation_steps,
        optim=self.training_parameters.optimizer,
        learning_rate=self.training_parameters.lr,
        lr_scheduler_type=self.training_parameters.lr_scheduler,
        save_strategy=self.training_parameters.save_strategy,
        logging_steps=self.training_parameters.logging_steps,
        num_train_epochs=self.training_parameters.num_train_epoch,
        max_steps=self.training_parameters.max_steps,
        fp16=self.training_parameters.fp16,
        #push_to_hub=self.training_parameters.push_to_hub,
        )

        return training_arguments
    
    def Start_Traning_Model(self):
        torch.cuda.empty_cache()
        
        dataset = self.Load_dataset()
        model = self.Base_model()
        tokenizer = self.Zephyr_tokenizer()
        peft_config= self.peft_Lora_configuration()
        model = get_peft_model(model, peft_config)
        training_arguments = self.Training_Argument()

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.training_parameters.max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=self.training_parameters.packing,

           )

        trainer.train()

        trainer.model.save_pretrained(self.model_training_config.model_name)

        """ del model
        del trainer
        import gc
        gc.collect()
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(
        self.model_training_config.model_check_point,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
        )
        model = PeftModel.from_pretrained(base_model, self.config.model_name)
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_check_point, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.save_pretrained(self.config.tokenizer_name) """