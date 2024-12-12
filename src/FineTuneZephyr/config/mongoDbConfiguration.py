from src.FineTuneZephyr.constants import *
from src.FineTuneZephyr.logging import logger
from src.FineTuneZephyr.utils.common import read_yaml,check_mongo_server_connection
from src.FineTuneZephyr.entity.server_entity import Mongo_DB_Server
from dotenv import load_dotenv
import os

load_dotenv()
class ServerConfigurationManager:
    def __init__(self, SrvrCnfg = SERVER_CONFIG_FILE_PATH):
        self.Sconfig = read_yaml(SrvrCnfg)
        self.Musername= os.getenv("mongo_username")
        self.Mpassword = os.getenv("mongo_password")
        


    def return_mongoconfiguration(self)-> Mongo_DB_Server:

        URI = f"mongodb+srv://{self.Musername}:{self.Mpassword}@cluster0.wnkizzs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0""

        # mongo_username=os.getenv("mongo_username")
        # mongo_password=os.getenv("mongo_password")
        if check_mongo_server_connection(URI):
            mongo_db_sever_variables = Mongo_DB_Server(

                mongodb_url=URI,
                mongodb_name=self.Sconfig.MONGODB.mongodb_name,
                mongodb_collection_name=self.Sconfig.MONGODB.mongodb_collection_name

            )
            return mongo_db_sever_variables
        else :
            print("Connection not established +++++====  Return From ServerConfigManger ++++++======")    

        