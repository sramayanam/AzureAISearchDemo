import requests
from PIL import Image
from io import BytesIO
import os
import requests
import base64
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


load_dotenv()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = (
    os.getenv("AZURE_OPENAI_ENDPOINT")
    + "openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
)
VISION_API_KEY = os.getenv("AZURE_AI_VISION_API_KEY")
VISION_ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT")


def analyze_image_with_azure_vision(image_url):
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT, credential=AzureKeyCredential(VISION_API_KEY)
    )

    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.READ,
            VisualFeatures.DENSE_CAPTIONS,
        ],
        gender_neutral_caption=True,
    )

    return result


def analyze_image_llm(image_url):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    response = requests.get(image_url)
    if response.status_code == 200:
        encoded_image = base64.b64encode(response.content).decode("ascii")
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that helps people caption the input image. Analyze the image and try to provide information about the characteristics of package, possible contents.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "\n"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                        {"type": "text", "text": "\n"},
                    ],
                },
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800,
        }

        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")

        if response.status_code == 200:
            return response.json()  # Assuming the response is in JSON format
        else:
            return {"error": "Failed to analyze the image"}
    else:
        return {"error": "Failed to fetch the image from the URL"}


