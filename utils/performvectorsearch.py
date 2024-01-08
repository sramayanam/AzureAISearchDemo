from re import search
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery
)
from dotenv import load_dotenv
import os
from utils.initializeclients import initializeclients
from utils.generateembeddings import generateembeddings

class performvectorsearch:
    def __init__(self): 
            self.client = initializeclients()

    def searchvector(self,query: str,searchfield: str,selectfields:list) -> list:
        openaiclient = self.client.create_azureopenai_client()
        ge = generateembeddings(query=query,openaiclient=openaiclient)
        vector_query = VectorizedQuery(vector=ge.getembedding(), k_nearest_neighbors=3, fields=searchfield)
        search_client = self.client.create_search_client()
        results = search_client.search(  
            search_text=query,  
            vector_queries= [vector_query],
            select=selectfields,
            filter="category eq 'Analytics'",
            query_type=QueryType.SEMANTIC, semantic_configuration_name='demo-semantic-config', query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
            top=3
        )  
        return results


