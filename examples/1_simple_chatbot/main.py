import os
from dotenv import load_dotenv

# Clients
from datapizza.clients.google import GoogleClient
from datapizza.clients.openai import OpenAIClient

# To build a chat we need Memory components
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

load_dotenv()

# OpenAI
# client = OpenAIClient(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model = "gpt-5",
#     system_prompt = "You are a helpful assistant that answers questions about the world."
# )


# Google Gemini
client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model = "gemini-flash-latest",
    system_prompt = "You are a helpful assistant that answers questions about the world."
)

memory = Memory()

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    
    response = client.invoke(user_input, memory=memory) # !IMPORTANT pass memory to the client
    
    print(f"Assistant: {response.text}")
    
    memory.add_turn(TextBlock(content=user_input), role=ROLE.USER) # add user input to memory
    memory.add_turn(response.content, role=ROLE.ASSISTANT) # add assistant response to memory