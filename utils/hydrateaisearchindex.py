import os  
import json  
from openai import AzureOpenAI
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchIndexingBufferedSender  
from utils.initializeclients import initializeclients
from utils.generateembeddings import generateembeddings
from utils.bootstrap import bootstrap

class hydrateaisearchindex:
    def __init__(self): 
            self.client = initializeclients()
            self.openaiclient = self.client.create_azureopenai_client()
            self.searchclient = self.client.create_search_client()
            self.searchendpoint = bootstrap().searchendpoint
            self.searchkey = bootstrap().searchkey
            self.searchindexname = bootstrap().searchindexname

    def loadsampledata(self):
        with open('./sample.json', 'r', encoding='utf-8') as file:
            return json.load(file)

    def createvectors(self,input_data):
        for item in input_data:
            title = item['title']
            content = item['content']
            getitle = generateembeddings(query=title,openaiclient=self.openaiclient)
            gecontent = generateembeddings(query=content,openaiclient=self.openaiclient)
            title_embeddings = getitle.getembedding()
            content_embeddings = gecontent.getembedding()
            item['titleVector'] = title_embeddings
            item['contentVector'] = content_embeddings
        with open("./docVectors.json", "w") as f:
            json.dump(input_data, f)

    @staticmethod
    def hydrateindex(self):
        self.createvectors(self.loadsampledata())
        with open('./docVectors.json', 'r') as file:  
            documents = json.load(file) 
        serviceendpoint = self.searchendpoint 
        searchkey = self.searchkey    
        credential = AzureKeyCredential(searchkey) 
        with SearchIndexingBufferedSender(  
            endpoint=serviceendpoint,  
            index_name=self.searchindexname,  
            credential=credential,  
        ) as batch_client:  
            batch_client.upload_documents(documents=documents)  
        print(f"Uploaded {len(documents)} documents in total")

