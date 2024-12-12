import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)


import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from src.FineTuneZephyr.exception import FineTuneZephyrException 
from src.FineTuneZephyr.logging import logger

class FineTuneZephyrDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise FineTuneZephyrException(e, sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise FineTuneZephyrException(e, sys)
    
    def pushing_data_to_mongodb(self,record,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise FineTuneZephyrException(e,sys)
if __name__ == '__main__':
    FILE_PATH="./Training_data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    DATABASE="FineTuneZephyr_data"
    COLLECTION="chatbot_assistant"
    zephyrobj = FineTuneZephyrDataExtract()
    records = zephyrobj.csv_to_json_convertor(FILE_PATH)
    noofrecords = zephyrobj.pushing_data_to_mongodb(records,DATABASE,COLLECTION)
    print(noofrecords)