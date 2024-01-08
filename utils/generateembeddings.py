from utils.initializeclients import initializeclients
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt  

class generateembeddings:
    def __init__(self,query: str,openaiclient: AzureOpenAI): 
        self.embeddingmodel = "text-embedding-ada-002"
        self.openaiclient = openaiclient
        self.query = query

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
    def getembedding(self) -> list:  
        openaiclient = self.openaiclient
        embedding =  openaiclient.embeddings.create(  
            input=self.query,  
            model=self.embeddingmodel)
        return embedding.data[0].embedding