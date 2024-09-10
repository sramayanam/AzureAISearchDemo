import streamlit as st
import requests
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import os
import numpy as np
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def fetch_similarimages(vector):
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

        # Execute the SELECT query to fetch image links
        input_vector_str = str(vector)
        select_query = "SELECT urllink FROM vectors ORDER BY vector <=> %s LIMIT 2;"
        cursor.execute(select_query, (input_vector_str,)) 
        image_links = cursor.fetchall()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return [link[0] for link in image_links]
    except (Exception, psycopg2.Error) as error:
        st.error(f"Error fetching image links: {error}")
        return []


def vectorize(url, method, headers=None, data=None):
    try:
        response = requests.request(method, url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

uploaded_files = st.file_uploader(
    "Upload an image File :", accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    api_url = "https://eastus.api.cognitive.microsoft.com/computervision/retrieval:vectorizeImage?api-version=2024-02-01&model-version=2023-04-15"
    api_method = "POST"
    api_headers = {
    "Ocp-Apim-Subscription-Key": os.getenv("AZURE_AI_VISION_API_KEY"),
    "Content-Type": "application/octet-stream",
    }
    img_vector = vectorize(api_url, api_method, headers=api_headers, data=bytes_data)["vector"]
   
    st.write("filename:", uploaded_file.name)
    #st.write("VectorData:", img_vector)
    # Fetch image links from the database
    image_links = fetch_similarimages(img_vector)

    # Render the image links in the Streamlit app
    st.title("Image Links")
    for link in image_links:
        st.image(link, caption=link)
    