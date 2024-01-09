from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery
)
from utils.initializeclients import initializeclients
from utils.generateembeddings import generateembeddings

class performvectorsearch:
    """
    A class used to perform vector search operations.

    ...

    Attributes
    ----------
    client : object
        an instance of the initializeclients class

    Methods
    -------
    searchvector(query: str, searchfield: str, selectfields: list) -> list:
        Performs a vector search and returns the results.
    """

    def __init__(self): 
        """
        Constructs all the necessary attributes for the performvectorsearch object.

        ...

        Attributes
        ----------
        client : object
            an instance of the initializeclients class
        """
        self.client = initializeclients()

    def searchvector(self, query: str, searchfield: str, selectfields: list) -> list:
        """
        Performs a vector search and returns the results.

        Parameters
        ----------
        query : str
            the query to search
        searchfield : str
            the field to search in
        selectfields : list
            the fields to select in the results

        Returns
        -------
        list
            a list of the search results
        """
        try:
            openaiclient = self.client.create_azureopenai_client()
            ge = generateembeddings(query=query, openaiclient=openaiclient)
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
        except Exception as ex:
            print(f'Exception: {ex}')
            return []