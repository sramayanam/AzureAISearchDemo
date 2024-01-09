import os
from dotenv import load_dotenv

class bootstrap:
    """
    A class used to bootstrap the application.

    ...

    Attributes
    ----------
    searchendpoint : str
        the Azure AI Search URL
    searchkey : str
        the Azure AI Search API key
    openaiapiversion : str
        the version of the Azure OpenAI API
    openaiendpoint : str
        the Azure OpenAI endpoint
    openaikey : str
        the Azure OpenAI key
    searchindexname : str
        the name of the search index

    Methods
    -------
    __init__():
        Initializes the bootstrap object and loads environment variables.
    """

    def __init__(self):
        """
        Initializes the bootstrap object and loads environment variables.

        ...

        Attributes
        ----------
        searchendpoint : str
            the Azure AI Search URL
        searchkey : str
            the Azure AI Search API key
        openaiapiversion : str
            the version of the Azure OpenAI API
        openaiendpoint : str
            the Azure OpenAI endpoint
        openaikey : str
            the Azure OpenAI key
        searchindexname : str
            the name of the search index
        """
        load_dotenv()
        self.searchendpoint = os.getenv("AZURE_AISEARCH_URL")
        self.searchkey = os.getenv("AZURE_AISEARCH_API_KEY")
        self.openaiapiversion = "2023-12-01-preview"
        self.openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openaikey = os.getenv("AZURE_OPENAI_KEY")
        self.searchindexname=os.getenv("searchindexname")