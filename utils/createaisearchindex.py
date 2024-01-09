import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

"""
This script is used to create an Azure AI Search index with Vector Search and Semantic Search capabilities.
The script imports necessary modules and classes from the Azure SDK for Python. It uses the SearchIndexClient to interact with the Azure AI Search service, and the various models to define the structure of the index and its fields.
The script is designed to be run as a standalone script and does not contain any functions or classes. It uses environment variables to get the necessary credentials and settings for the Azure AI Search service.
"""

load_dotenv()
service_endpoint = os.getenv("AZURE_AISEARCH_URL")
key = os.getenv("AZURE_AISEARCH_API_KEY")
credential = AzureKeyCredential(key)
index_name = "demo-index"

index_client = SearchIndexClient(
    endpoint=service_endpoint, credential=credential)

"""
The index_fields list defines the fields that will be included in the index. 
Each field is defined using one of the field classes from the Azure SDK for Python, 
and the properties of the field are set using the parameters of the class constructor.
"""
index_fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchableField(name="category", type=SearchFieldDataType.String, filterable=True),
    SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="demoHnswProfile"),
    SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="demoHnswProfile"),
]

"""
The vector_search object defines the vector search settings for the index. 
It includes the algorithms and profiles to be used for vector search.
"""
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="demoHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE
            )
        ),
        ExhaustiveKnnAlgorithmConfiguration(
            name="demoExhaustiveKnn",
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
            parameters=ExhaustiveKnnParameters(
                metric=VectorSearchAlgorithmMetric.COSINE
            )
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="demoHnswProfile",
            algorithm_configuration_name="demoHnsw",
        ),
        VectorSearchProfile(
            name="demoExhaustiveKnnProfile",
            algorithm_configuration_name="demoExhaustiveKnn",
        )
    ]
)

"""
The semantic_config object defines the semantic search settings for the index. 
It includes the fields to be used for semantic search.
"""
semantic_config = SemanticConfiguration(
    name="demo-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        keywords_fields=[SemanticField(field_name="category")],
        content_fields=[SemanticField(field_name="content")]
    )
)

semantic_search = SemanticSearch(configurations=[semantic_config])

index = SearchIndex(name=index_name, fields=index_fields, vector_search=vector_search, semantic_search=semantic_search)

result = index_client.create_or_update_index(index)

print(f' {result.name} created')