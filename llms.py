from crewai import LLM
import os
from dotenv import load_dotenv

load_dotenv()

llama_scout = LLM(
    model="meta/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
    api_key=os.getenv("NVIDIA_NIM_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)
