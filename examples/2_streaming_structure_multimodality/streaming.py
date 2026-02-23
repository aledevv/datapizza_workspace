import os
from dotenv import load_dotenv

# Clients
from datapizza.clients.google import GoogleClient

load_dotenv()


# Google Gemini
client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model = "gemini-flash-latest",
    system_prompt = "You are a helpful AI assistant."
)

response = client.stream_invoke("Explain the theory of relativity in a sentence.")
    
# Streaming (it is an iterator) -> Better for user experience and for long responses
for chunk in response:
    print(chunk.delta, end="", flush=True)
