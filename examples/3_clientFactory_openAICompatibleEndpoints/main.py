import os
from dotenv import load_dotenv

# Clients
from datapizza.clients import ClientFactory
from datapizza.clients.factory import Provider

load_dotenv()


# Use multiple modalities as input like text, audio, images (depends on the model)

# Create any provider with the same interface
google = ClientFactory.create(
    provider=Provider.GOOGLE,
    api_key=os.getenv("GOOGLE_API_KEY"),
    model = "gemini-flash-latest",
    system_prompt = "You are a helpful AI assistant."
)

# openai = ClientFactory.create(
#     provider=Provider.OPENAI,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model = "gpt-5",
# )

# anthropic = ClientFactory.create(
#     provider=Provider.ANTHROPIC,
#     api_key=os.getenv("ANTHROPIC_API_KEY"),
#     model = "claude-2",
#     system_prompt = "You are a helpful AI assistant."
# )

response = google.invoke("Explain the theory of relativity in a sentence.")
print(response.text)

# ------------------------------- 

# OpenAI-like clients (follow README.md instructions to setup everything)
from datapizza.clients.openai_like import OpenAILikeClient

# Create client for Ollama
client = OpenAILikeClient(
    api_key="", # Ollama doesn't need API key, but we need to pass it to follow the same interface as other clients
    model="qwen3:4b", # The name of the model in Ollama
    system_prompt="You are a helpful AI assistant."
    base_url="http://localhost:11434/v1" # ! The URL of your Ollama server
)
    
response = client.invoke("Explain the theory of relativity in a sentence.")
print(response.text)

# ! Check datapizza-ai documentation for further instructions