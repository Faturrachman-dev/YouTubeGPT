from langchain_nvidia_ai_endpoints import ChatNVIDIA

client = ChatNVIDIA(
  model="meta/llama-3.1-405b-instruct",
  api_key="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", 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

for chunk in client.stream([{"role":"user","content":"Write a limerick about the wonders of GPU computing."}]): 
  print(chunk.content, end="")

  
