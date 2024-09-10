import requests
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import os
import numpy as np
from dotenv import load_dotenv
import psycopg2


load_dotenv()

def insert_vectors(url_vector_pairs, batch_size=100):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )

        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()

        # Create the vectors table if it doesn't exist
        # CREATE EXTENSION vector;
        # CREATE EXTENSION azure_storage;
        create_table_query = """
        CREATE TABLE IF NOT EXISTS vectors (
            id SERIAL PRIMARY KEY,
            urllink TEXT,
            vector vector(1024)
        );
        """
        cursor.execute(create_table_query)
        conn.commit()

        # Insert the vectors into the table in batches
        insert_query = "INSERT INTO vectors (urllink, vector) VALUES (%s, %s)"
        for i in range(0, len(url_vector_pairs), batch_size):
            batch = url_vector_pairs[i : i + batch_size]
            cursor.executemany(insert_query, batch)
            conn.commit()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        print("Vectors inserted successfully.")
    except (Exception, psycopg2.Error) as error:
        print(f"Error inserting vectors: {error}")


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def generate_sas_links(sas_expiry_hours=24):
    try:

        blob_service_client = BlobServiceClient.from_connection_string(
            os.getenv("BLOB_CONNECTION_STRING")
        )

        container_client = blob_service_client.get_container_client(
            os.getenv("BLOB_CONTAINER_NAME")
        )

        key = blob_service_client.credential.account_key
        storage_account_name = blob_service_client.account_name
        container_name = os.getenv("BLOB_CONTAINER_NAME")

        blobs = container_client.list_blobs()

        sas_links = []
        for blob in blobs:
            blob_sas = generate_blob_sas(
                account_name=storage_account_name,
                container_name=container_name,
                blob_name=blob.name,
                account_key=key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=sas_expiry_hours),
            )
            image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob.name}?{blob_sas}"
            sas_links.append(image_url)

        return sas_links
    except Exception as e:
        print(f"An error occurred: {e}")


def vectorize(url, method, headers=None, data=None):
    try:
        response = requests.request(method, url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


sas_links = generate_sas_links(24)

url_vector_pairs = []

for sas_link in sas_links:
    print(sas_link)
    api_url = "https://eastus.api.cognitive.microsoft.com/computervision/retrieval:vectorizeImage?api-version=2024-02-01&model-version=2023-04-15"
    api_method = "POST"
    api_headers = {
        "Ocp-Apim-Subscription-Key": os.getenv("AZURE_AI_VISION_API_KEY"),
        "Content-Type": "application/json",
    }
    data = {"url": sas_link}
    img_vector = vectorize(api_url, api_method, headers=api_headers, data=data)[
        "vector"
    ]

    api_url = "https://eastus.api.cognitive.microsoft.com/computervision/retrieval:vectorizeText?api-version=2024-02-01&model-version=2023-04-15"
    api_method = "POST"
    api_headers = {
        "Ocp-Apim-Subscription-Key": os.getenv("AZURE_AI_VISION_API_KEY"),
        "Content-Type": "application/json",
    }
    data = {
        "text": "Box Wrapped Compressor Kit. TRUPER 116 PSI product. An Air Compressor Package. Also has gravity feed air gun. LVMP The package includes 50 Litre lubricator, a 5 meter connector and a pressure hose"
    }
    text_vector = vectorize(api_url, api_method, headers=api_headers, data=data)[
        "vector"
    ]
    print(img_vector)
    print(text_vector)
    print(
        "Cosine similarity between image and text vectors: ",
        cosine_similarity(img_vector, text_vector),
    )

    url_vector_pairs.append((sas_link, img_vector))


insert_vectors(url_vector_pairs, batch_size=10)
