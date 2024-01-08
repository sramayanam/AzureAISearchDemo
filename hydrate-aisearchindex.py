
import os  
import json  
from openai import AzureOpenAI
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchIndexingBufferedSender  

def load_sample_data():
    with open('./sample.json', 'r', encoding='utf-8') as file:
        return json.load(file)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(2))
def generate_embeddings(text):
    client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version = "2023-05-15",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    model: str = "text-embedding-ada-002" 
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def createvectors(input_data):
    for item in input_data:
        title = item['title']
        content = item['content']
        title_embeddings = generate_embeddings(title)
        content_embeddings = generate_embeddings(content)
        item['titleVector'] = title_embeddings
        item['contentVector'] = content_embeddings
        # Output embeddings to docVectors.json file
    with open("./docVectors.json", "w") as f:
        json.dump(input_data, f)

def hydrate_index():
    service_endpoint = os.getenv("AZURE_AISEARCH_URL") 
    key = os.getenv("AZURE_AISEARCH_API_KEY")    
    credential = AzureKeyCredential(key)
    index_name = "demo-index"
    with open('./docVectors.json', 'r') as file:  
        documents = json.load(file)  
    with SearchIndexingBufferedSender(  
        endpoint=service_endpoint,  
        index_name=index_name,  
        credential=credential,  
    ) as batch_client:  
        batch_client.upload_documents(documents=documents)  
    print(f"Uploaded {len(documents)} documents in total")

def main():
    load_dotenv()  
    createvectors(load_sample_data())
#    hydrate_index()

if __name__ == "__main__":
    main()