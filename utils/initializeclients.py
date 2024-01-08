
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from utils.bootstrap import bootstrap

class initializeclients:
    def __init__(self):
        try:
            self.searchendpoint = bootstrap().searchendpoint
            self.searchkey = bootstrap().searchkey
            self.openaikey = bootstrap().openaikey
            self.openaiendpoint = bootstrap().openaiendpoint
            self.openaiapiversion = bootstrap().openaiapiversion
            self.searchindexname  = bootstrap().searchindexname     
        except Exception as e:
            print(f"Error occurred during initialization: {e}")


    def create_search_client(self) -> SearchClient:
        try:
            return SearchClient(self.searchendpoint, self.searchindexname, credential=AzureKeyCredential(self.searchkey))
        except Exception as e:
            print(f"Error occurred during creation of the search client: {e}")


    def create_azureopenai_client(self) -> AzureOpenAI:
        try:
            return AzureOpenAI(azure_endpoint=self.openaikey, api_version=self.openaiapiversion, api_key=self.openaikey)
        except Exception as e:
            print(f"Error occurred during creation of the Azure OpenAI client: {e}")
