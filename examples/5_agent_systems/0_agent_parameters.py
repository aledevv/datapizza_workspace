import os
from dotenv import load_dotenv

# Clients
from datapizza.clients.google import GoogleClient
from datapizza.tools import tool
from datapizza.agents import Agent

load_dotenv()


# Google Gemini
client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model = "gemini-2.5-flash",     # use 2.5 flash since from 3 (and surely 3.1) it uses tought signature that is required to be sent back to google together with response (so if you use tools check if your model needs this)
    system_prompt = "You are a helpful AI assistant."
)


@tool
def get_weather(city: str, when: str) -> str:
    """Get the current weather for a city at a specific time."""
    return f"The current weather in {city} on {when} is sunny and 25 degrees Celsius."


# let's see other parameters for agents
weather_agent = Agent(
    name="weather_agent",  # just for debugging purposes
    client=client,
    system_prompt="You are a helpful weather assistant.",
    tools=[get_weather],
    # terminate_on_text=False, # means that agent doesn't terminate working when it provides textual response
    # max_steps=3, # agent works for 3 steps (3 actions), to avoid getting in a loop, but also to make it use a certain max number of tools
)


weather_agent.run(
    "What is the weather in Milan tomorrow?",
    tool_choice="required", # it is required to use a tool, you can also do #! tool_choice=["get_weather", "some_other_tool"] to force selecting from a list of tools
)


