from utils.initializeclients import initializeclients
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt  

class generateembeddings:
    """
    A class used to generate embeddings for a given query using Azure OpenAI.

    ...

    Attributes
    ----------
    embeddingmodel : str
        the name of the embedding model to use
    openaiclient : AzureOpenAI
        an instance of the Azure OpenAI client
    query : str
        the query to generate embeddings for

    Methods
    -------
    __init__(query: str, openaiclient: AzureOpenAI):
        Initializes the generateembeddings object with the given query and Azure OpenAI client.
    getembedding() -> list:
        Generates and returns an embedding for the query.
    """

    def __init__(self,query: str,openaiclient: AzureOpenAI): 
        """
        Initializes the generateembeddings object with the given query and Azure OpenAI client.

        Parameters
        ----------
        query : str
            the query to generate embeddings for
        openaiclient : AzureOpenAI
            an instance of the Azure OpenAI client
        """
        try:
            self.embeddingmodel = "text-embedding-ada-002"
            self.openaiclient = openaiclient
            self.query = query
        except Exception as e:
            print(f"Error occurred during initialization: {e}")

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
    def getembedding(self) -> list:  
        """
        Generates and returns an embedding for the query.

        Returns
        -------
        list
            a list containing the embedding for the query
        """
        try:
            openaiclient = self.openaiclient
            embedding =  openaiclient.embeddings.create(  
                input=self.query,  
                model=self.embeddingmodel)
            return embedding.data[0].embedding
        except Exception as e:
            print(f"Error occurred during generating embeddings: {e}")