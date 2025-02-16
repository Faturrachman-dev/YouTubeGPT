import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load environment variables from .env file
load_dotenv()

client = ChatNVIDIA(
  model="meta/llama-3.1-405b-instruct",
  api_key=os.getenv("NVIDIA_API_KEY"),
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

for chunk in client.stream([{"role":"user","content":"Write a limerick about the wonders of GPU computing."}]): 
  print(chunk.content, end="")

  
