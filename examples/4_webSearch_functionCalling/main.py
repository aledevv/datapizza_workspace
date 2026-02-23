import os
from dotenv import load_dotenv

# Clients
from datapizza.clients.google import GoogleClient
from datapizza.tools import tool
from datapizza.agents import Agent

load_dotenv()

# let's define some tools

@tool   # use decorator to define a tool
def add(a: float | int, b: float | int) -> float:
    """Add two numbers together."""
    return a + b


# Google Gemini
client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model = "gemini-2.5-flash",     # use 2.5 flash since from 3 (and surely 3.1) it uses tought signature that is required to be sent back to google together with response (so if you use tools check if your model needs this)
    system_prompt = "You are a helpful AI assistant."
)

response = client.invoke(
    "What is 2 + 6?",
    tools=[add]   # we can pass the tools to the model, so it can use them if needed
)

print(response.function_calls)   # we can check if the model called any tools

# -----------------------------------------

@tool
def get_weather(city: str, when: str) -> str:
    """Get the current weather for a city at a specific time."""
    return f"The current weather in {city} on {when} is sunny and 25 degrees Celsius."


# let's build a weather agent
weather_agent = Agent(
    name="weather_agent",  # just for debugging purposes
    client=client,
    system_prompt="You are a helpful weather assistant.",
    tools=[get_weather],
)


weather_agent.run("What is the weather in Milan tomorrow?") # it displays a debug trace showing different steps of the agent, chosen tools (and their output) and the final answer is: "The current weather in Milan on tomorrow is sunny and 25 degrees Celsius." (since we are using a dummy function)


# Web search tool (DuckDuckGo)

from datapizza.tools.duckduckgo import DuckDuckGoSearchTool

web_search_agent = Agent(
    name="web_search_agent",
    client=client,
    system_prompt="You are a helpful assistant that can search the web for information.",
    tools=[DuckDuckGoSearchTool()],
)

response = web_search_agent.run("How tall is the Duomo of Milan?")

# You can check in the terminal what the agent used as query to search on duckduckgo 
# together with the information retrieved from the search, and the FINAL ANSWER (that can be way shoerter w.r.t. the retrieved information)