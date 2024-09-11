import streamlit as st
import os
from fetchsimimgimgs import fetch_similarimages
from fetchsimimgimgs import vectorize
from fetchsimtxtimgs import vectorize as vectorize_txt
from fetchsimtxtimgs import fetch_similarimages as txt_fetch_similarimages

def fetch_images_based_on_text():
    api_url = "https://eastus.api.cognitive.microsoft.com/computervision/retrieval:vectorizeText?api-version=2024-02-01&model-version=2023-04-15"
    api_method = "POST"
    api_headers = {
            "Ocp-Apim-Subscription-Key": os.getenv("AZURE_AI_VISION_API_KEY"),
            "Content-Type": "application/json",
        }
    data = {"text": st.session_state.text_input}
    text_vector = vectorize_txt(api_url, api_method, headers=api_headers, data=data)["vector"]
    image_links = txt_fetch_similarimages(text_vector)
    st.session_state.image_links = image_links
    st.session_state.images_fetched = True
    

pages = {
    "Fetch Similar Images by Image": "img",
    "Fetch Similar Images by Text": "txt",
}

selected_page = st.sidebar.selectbox("Perform Search", options=list(pages.keys()))

if pages[selected_page] == "img":

    st.title("Fetch Similar Images Based on an Image")        
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
        st.session_state.images_fetched = True

elif pages[selected_page] == "txt":

    if 'images_fetched' not in st.session_state:
            st.session_state.images_fetched = False
    if 'image_links' not in st.session_state:
            st.session_state.image_links = []
    if not st.session_state.images_fetched:
        st.title("Welcome to Vector Search !!!! Enter Text to test drive")
        text_input = st.text_input("Enter text to find similar images:")
        if text_input:
            st.session_state.text_input = text_input
            st.button('Find Images', on_click=fetch_images_based_on_text)
    else:
        st.title("Found these images for you !!!!! ")
        for link in st.session_state.image_links:
            st.image(link, caption=link)
