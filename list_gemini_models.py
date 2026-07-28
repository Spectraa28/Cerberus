# list_gemini_models.py
import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name)