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