from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from utils.bootstrap import bootstrap

class initializeclients:
    """
    A class used to initialize clients for Azure Search and Azure OpenAI.

    ...

    Attributes
    ----------
    searchendpoint : str
        the Azure AI Search URL
    searchkey : str
        the Azure AI Search API key
    openaikey : str
        the Azure OpenAI key
    openaiendpoint : str
        the Azure OpenAI endpoint
    openaiapiversion : str
        the version of the Azure OpenAI API
    searchindexname : str
        the name of the search index

    Methods
    -------
    __init__():
        Initializes the initializeclients object and loads environment variables.
    create_search_client() -> SearchClient:
        Creates and returns an Azure Search client.
    create_azureopenai_client() -> AzureOpenAI:
        Creates and returns an Azure OpenAI client.
    """

    def __init__(self):
        """
        Initializes the initializeclients object and loads environment variables.

        ...

        Attributes
        ----------
        searchendpoint : str
            the Azure AI Search URL
        searchkey : str
            the Azure AI Search API key
        openaikey : str
            the Azure OpenAI key
        openaiendpoint : str
            the Azure OpenAI endpoint
        openaiapiversion : str
            the version of the Azure OpenAI API
        searchindexname : str
            the name of the search index
        """
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
        """
        Creates and returns an Azure Search client.

        Returns
        -------
        SearchClient
            an instance of the Azure Search client
        """
        try:
            return SearchClient(self.searchendpoint, self.searchindexname, credential=AzureKeyCredential(self.searchkey))
        except Exception as e:
            print(f"Error occurred during creation of the search client: {e}")

    def create_azureopenai_client(self) -> AzureOpenAI:
        """
        Creates and returns an Azure OpenAI client.

        Returns
        -------
        AzureOpenAI
            an instance of the Azure OpenAI client
        """
        try:
            return AzureOpenAI(azure_endpoint=self.openaikey, api_version=self.openaiapiversion, api_key=self.openaikey)
        except Exception as e:
            print(f"Error occurred during creation of the Azure OpenAI client: {e}")