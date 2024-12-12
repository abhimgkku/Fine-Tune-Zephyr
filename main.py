from src.FineTuneZephyr.logging import logger
from src.FineTuneZephyr.pipeline.Data_ingestion_pipeline import DataIngestionPipeline
#from src.FineTuneLlama2.pipeline.Data_validation_pipeline import DataValidationPipeline
#from src.FineTuneLlama2.pipeline.Data_transformation_pipeline import DataTransformationPipeline
#from src.FineTuneLlama2.pipeline.Model_training_pipeline import ModelTrainingPieline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e