import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery
)

def get_embedding(query: str) -> list:  
    openai_client = create_azureopenai_client()
    model = "text-embedding-ada-002"  
    embedding = openai_client.embeddings.create(  
        input=query,  
        model=model  )
    return embedding.data[0].embedding

def create_search_client(index_name: str) -> SearchClient:  
    service_endpoint = os.getenv("AZURE_AISEARCH_URL")  
    key = os.getenv("AZURE_AISEARCH_API_KEY") 
    index_name = index_name
    return SearchClient(service_endpoint, index_name, credential=AzureKeyCredential(key) )

def create_azureopenai_client() -> AzureOpenAI:

    return AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2023-12-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"))

def search_vector(query: str,searchfield: str,selectfields:list,index_name:str) -> list:
    vector_query = VectorizedQuery(vector=get_embedding(query), k_nearest_neighbors=3, fields=searchfield)
    search_client = create_search_client(index_name)
    results = search_client.search(  
        search_text=query,  
        vector_queries= [vector_query],
        select=selectfields,
        filter="category eq 'Analytics'",
        query_type=QueryType.SEMANTIC, semantic_configuration_name='demo-semantic-config', query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
        top=3
    )  
    return results

def main() -> None:  
    load_dotenv()  

    query="what is more comprehensive data tool?"
    results = search_vector(query,"contentVector,titleVector",["title", "content", "category"],"demo-index")


    for result in results:  
        print(f"Semantic Captions: {result['@search.captions'][0].highlights}")
        print(f"Semantic Text: {result['@search.captions'][0].text}")
        print(f"Title: {result['title']}")  
        print(f"Score: {result['@search.score']}")  
        print(f"ReRanker Score: {result['@search.reranker_score']}")  
        print(f"Content: {result['content']}")  
        print(f"Category: {result['category']}\n")  

if __name__ == "__main__":  
    main()

