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
    """
    A class used to hydrate the Azure AI Search index with data.

    ...

    Attributes
    ----------
    client : object
        an instance of the initializeclients class
    openaiclient : object
        an instance of the Azure OpenAI client
    searchclient : object
        an instance of the Azure Search client
    searchendpoint : str
        the Azure AI Search URL
    searchkey : str
        the Azure AI Search API key
    searchindexname : str
        the name of the search index

    Methods
    -------
    __init__():
        Initializes the hydrateaisearchindex object and loads environment variables.
    loadsampledata() -> dict:
        Loads sample data from a JSON file and returns it.
    createvectors(input_data: dict):
        Creates vector embeddings for the title and content of each item in the input data.
    """

    def __init__(self): 
        """
        Initializes the hydrateaisearchindex object and loads environment variables.

        ...

        Attributes
        ----------
        client : object
            an instance of the initializeclients class
        openaiclient : object
            an instance of the Azure OpenAI client
        searchclient : object
            an instance of the Azure Search client
        searchendpoint : str
            the Azure AI Search URL
        searchkey : str
            the Azure AI Search API key
        searchindexname : str
            the name of the search index
        """
        try:
            self.client = initializeclients()
            self.openaiclient = self.client.create_azureopenai_client()
            self.searchclient = self.client.create_search_client()
            self.searchendpoint = bootstrap().searchendpoint
            self.searchkey = bootstrap().searchkey
            self.searchindexname = bootstrap().searchindexname
        except Exception as e:
            print(f"Error occurred during initialization: {e}")

    def loadsampledata(self) -> dict:
        """
        Loads sample data from a JSON file and returns it.

        Returns
        -------
        dict
            a dictionary containing the sample data
        """
        try:
            with open('./sample.json', 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print("The file 'sample.json' was not found.")
        except Exception as e:
            print(f"Error occurred during loading sample data: {e}")

    def createvectors(self,input_data: dict):
        """
        Creates vector embeddings for the title and content of each item in the input data.

        Parameters
        ----------
        input_data : dict
            a dictionary containing the input data
        """
        try:
            for item in input_data:
                title = item['title']
                content = item['content']
                getitle = generateembeddings(query=title,openaiclient=self.openaiclient)
                gecontent = generateembeddings(query=content,openaiclient=self.openaiclient)
                title_embeddings = getitle.getembedding()
                content_embeddings = gecontent.getembedding()
                item['titleVector'] = title_embeddings
        except Exception as e:
            print(f"Error occurred during creating vectors: {e}")

    def hydrateindex(self):
        """
        Hydrates the Azure AI Search index with vector embeddings.

        This method first calls the createvectors method to generate vector embeddings for the title and content of each item in the sample data. It then loads the vector embeddings from a JSON file and hydrates the Azure AI Search index with these embeddings.
        """
        try:
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
        except FileNotFoundError:
            print("The file 'docVectors.json' was not found.")
        except Exception as e:
            print(f"Error occurred during hydrating index: {e}")
