import os
from dotenv import load_dotenv
class bootstrap:
    def __init__(self):
        load_dotenv()
        self.searchendpoint = os.getenv("AZURE_AISEARCH_URL")
        self.searchkey = os.getenv("AZURE_AISEARCH_API_KEY")
        self.openaiapiversion = "2023-12-01-preview"
        self.openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openaikey = os.getenv("AZURE_OPENAI_KEY")
        self.searchindexname=os.getenv("searchindexname")