# 📚 Datapizza AI Documentation

## Table of Contents

- [🏠 Datapizza AI (Home)](#0)

### Guides
- **Clients**
  - [Quick Start](#1)
  - [Multimodality](#2)
  - [Structured Responses](#3)
  - [Streaming](#4)
  - [Tools](#5)
  - [Real example: Chatbot](#6)
  - [Running with Ollama](#7)
- **Agents**
  - [Build your first agent](#8)
  - [Model Context Protocol (MCP)](#9)
- **RAG**
  - [Build a RAG](#10)
- **Pipeline**
  - [Ingestion Pipeline](#11)
  - [DagPipeline](#12)
  - [Functional Pipeline](#13)
- **Monitoring**
  - [Tracing](#14)
  - [Log level](#15)

### API Reference
- **[Clients](#17)**
  - [Client Factory](#18)
  - [Response](#24)
  - [Cache](#25)
  - **Avaiable Clients**
    - [Openai](#19)
    - [Google](#21)
    - [Anthropic](#20)
    - [Mistral](#22)
    - [Openai like](#23)
- **Agents**
  - [Agent](#16)
- **Embedders**
  - [ChunkEmbedder](#26)
  - [CohereEmbedder](#27)
  - [FastEmbedder](#28)
  - [GoogleEmbedder](#29)
  - [OllamaEmbedder](#30)
  - [OpenAIEmbedder](#31)
- **Vectorstore**
  - [Milvus](#65)
  - [Qdrant](#64)
- **[Memory](#32)**
- **Type**
  - [Blocks](#59)
  - [Chunk](#60)
  - [Media](#61)
  - [Node](#62)
  - [Tool](#63)
- **Pipelines**
  - [Dag](#52)
  - [Functional](#53)
  - [Ingestion](#51)
- **[Modules](#33)**
  - **[Parsers](#35)**
    - [TextParser](#36)
    - [DoclingParser](#37)
    - [AzureParser](#38)
  - [Treebuilder](#50)
  - [Captioners](#34)
  - **[Splitters](#44)**
    - [RecursiveSplitter](#47)
    - [TextSplitter](#48)
    - [NodeSplitter](#45)
    - [PDFImageSplitter](#46)
  - [Metatagger](#49)
  - [Rewriters](#43)
  - **[Rerankers](#40)**
    - [CohereReranker](#41)
    - [TogetherReranker](#42)
  - **Prompt**
    - [ChatPromptTemplate](#39)
- **Tools**
  - [MCPClient](#56)
  - [DuckDuckGo](#57)
  - [FileSystem](#55)
  - [SQLDatabase](#58)
  - [WebFetch](#54)

---

<a id="0"></a>

## Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/

# datapizza-ai

**Build reliable Gen AI solutions without overhead**

`datapizza-ai` provides clear interfaces and predictable behavior for agents and RAG. End-to-end visibility and reliable orchestration keep engineers in control from PoC to scale

## Installation

Install the library using pip:

```
pip install datapizza-ai
```

## Key Features

* **Integration with AI Providers**: Seamlessly connect with AI services like OpenAI and Google VertexAI.
* **Complex workflows, minimal code.**: Design, automate, and scale powerful agent workflows without the overhead of boilerplate.
* **Retrieval-Augmented Generation (RAG)**: Enhance AI responses with document retrieval.
* **Faster delivery, easier onboarding for new engineers**: Rebuild a RAG + tools agent without multi-class plumbing; parity with simpler, typed interfaces.
* **Up to 40% less debugging time**: Trace and log every LLM/tool call with inputs/outputs

## Quick Start

To get started with `datapizza-ai`, ensure you have Python `>=3.10.0,<3.13.0` installed.

Here's a basic example demonstrating how to use agents in `datapizza-ai`:

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

@tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

client = OpenAIClient(api_key="YOUR_API_KEY")
agent = Agent(name="assistant", client=client, tools = [get_weather])

response = agent.run("What is the weather in Rome?")
# output: The weather in Rome is sunny
```

---


# 📘 GUIDES


## Clients

<a id="1"></a>

## Quick Start - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Clients/quick_start/

# Quick Start

This guide will help you get started with the `OpenAIClient` in datapizza-ai. For specialized topics, check out our detailed guides on [multimodality](../multimodality/), [streaming](../streaming/) and [building chatbots](../chatbot/).

## Installation

First, make sure you have datapizza-ai installed:

```
pip install datapizza-ai
```

## Basic Setup

```
from datapizza.clients.openai import OpenAIClient

# Initialize the client with your API key
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4o-mini",  # Default model
    system_prompt="You are a helpful assistant",  # Optional
    temperature=0.7  # Optional, controls randomness (0-2)
)
```

```
# Basic text response
response = client.invoke("What is the capital of France?")
print(response.text)
# Output: "The capital of France is Paris."
```

## Core Methods

```
response = client.invoke(
    input="Explain quantum computing in simple terms",
    temperature=0.5,  # Override default temperature
    max_tokens=200,   # Limit response length
    system_prompt="You are a physics teacher"  # Override system prompt
)

print(response.text)
print(f"Tokens used: {response.completion_tokens_used}")
```

## Async invoke

```
import asyncio

async def main():
    return await client.a_invoke(
        input="Explain quantum computing in simple terms",
        temperature=0.5,  # Override default temperature
        max_tokens=200,   # Limit response length
        system_prompt="You are a physics teacher"  # Override system prompt
    )

response = asyncio.run(main())

print(response.text)
print(f"Tokens used: {response.completion_tokens_used}")
```

## Working with Memory

Memory allows you to maintain conversation context:

```
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

memory = Memory()

# First interaction
response1 = client.invoke("My name is Alice", memory=memory)
memory.add_turn(TextBlock(content="My name is Alice"), role=ROLE.USER)
memory.add_turn(response1.content, role=ROLE.ASSISTANT)

# Second interaction - the model remembers Alice
response2 = client.invoke("What's my name?", memory=memory)
print(response2.text)  # Should mention Alice
```

## Token Management

Monitor your usage:

```
response = client.invoke("Explain AI")
print(f"Tokens used: {response.completion_tokens_used}")
print(f"Prompt token used: {response.prompt_tokens_used}")
print(f"Cached token used: {response.cached_tokens_used}")
```

That's it! You're ready to start building with the OpenAI client. Check out the specialized guides above for advanced features and patterns.

## What's Next?

Now that you know the basics, explore our specialized guides:

### 📸 [Multimodality Guide](../multimodality/)

Work with images, PDFs, and other media types for visual AI applications.

### 🌊 [Streaming Guide](../streaming/)

Build responsive applications with real-time text generation and streaming.

### 🛠️ [Tools Guide](../tools/)

Extend AI capabilities by integrating external functions and tools.

### 📊 [Structured Responses Guide](../structured_responses/)

Work with strongly-typed outputs using JSON schemas and Pydantic models.

### 🤖 [Chatbot Guide](../chatbot/)

Create sophisticated conversational AI with memory and context management.

---

<a id="2"></a>

## Multimodality - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Clients/multimodality/

# Multimodality

The clients supports various media types including images and PDFs, allowing you to create rich multimodal applications.

## Supported Media Types

| Media Type | Supported Formats | Source Types |
| --- | --- | --- |
| Images | PNG, JPEG, GIF, WebP | File path, URL, base64 |
| PDFs | PDF documents | File path, base64 |

## Basic Image Input

### Single Image from File

```
from datapizza.clients.openai import OpenAIClient
from datapizza.type import Media, MediaBlock, TextBlock

client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o"  # Vision models required for images
)

# Create image media object
image = Media(
    media_type="image",
    source_type="path",
    source="image.png", # Use the correct path
    extension="png"
)

# Create media block
media_block = MediaBlock(media=image)
text_block = TextBlock(content="What do you see in this image?")

# Send multimodal input
response = client.invoke(
    input=[text_block, media_block],
    max_tokens=200
)

print(response.text)
```

### Image from URL

```
# Image from URL
image_url = Media(
    media_type="image",
    source_type="url",
    source="https://example.com/image.png",
    extension="png"
)

response = client.invoke(
    input=[
        TextBlock(content="Describe this image"),
        MediaBlock(media=image_url)
    ]
)
print(response.text)
```

### Image from Base64

```
import base64

# Read and encode image
with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

image_b64 = Media(
    media_type="image",
    source_type="base64",
    source=base64_image,
    extension="png"
)

response = client.invoke(
    input=[
        TextBlock(content="Analyze this image"),
        MediaBlock(media=image_b64)
    ]
)
print(response.text)
```

## Multiple Images

Compare or analyze multiple images in a single request:

```
# Multiple images for comparison
image1 = Media(
    media_type="image",
    source_type="path",
    source="before.png",
    extension="png"
)

image2 = Media(
    media_type="image",
    source_type="path",
    source="after.png",
    extension="png"
)

response = client.invoke(
    input=[
        TextBlock(content="Compare these two images and describe the differences"),
        MediaBlock(media=image1),
        MediaBlock(media=image2)
    ],
    max_tokens=300
)

print(response.text)
```

## Working with PDFs

```
# PDF from file path
pdf_doc = Media(
    media_type="pdf",
    source_type="path",
    source="document.pdf",
    extension="pdf"
)

response = client.invoke(
    input=[
        TextBlock(content="Summarize the key points from this document"),
        MediaBlock(media=pdf_doc)
    ],
    max_tokens=500
)

print(response.text)
```

## Working with Audio

Google handle audio inline

```
pip install datapizza-ai-clients-google
```

```
from datapizza.clients.google import GoogleClient
from datapizza.type import Media, MediaBlock, TextBlock

client = GoogleClient(
    api_key="YOUR_API_KEY",
    model="gemini-2.0-flash-exp"
)
# PDF from file path
media = Media(
    media_type="audio",
    source_type="path",
    source="sample.mp3",
    extension="mp3"
)

response = client.invoke(
    input=[
        TextBlock(content="Summarize the key points from this audio file"),
        MediaBlock(media=media)
    ],
)

print(response.text)
```

---

<a id="3"></a>

## Structured Responses - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Clients/structured_responses/

# Structured Responses

Generate structured, typed data from AI responses using Pydantic models. This ensures consistent output format and enables easy data validation.

## Basic Usage

### Simple Model

```
from datapizza.clients.openai import OpenAIClient
from pydantic import BaseModel

client = OpenAIClient(api_key="your-api-key", model="gpt-4o-mini")

class Person(BaseModel):
    name: str
    age: int
    occupation: str

response = client.structured_response(
    input="Create a profile for a software engineer",
    output_cls=Person
)

person = response.structured_data[0]
print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")
```

### Complex Models

```
from typing import List
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: float = Field(gt=0, description="Price must be positive")
    tags: List[str]
    in_stock: bool

class Store(BaseModel):
    name: str
    location: str
    products: List[Product]

response = client.structured_response(
    input="Create a tech store with 3 products",
    output_cls=Store
)

store = response.structured_data[0]
print(f"Store: {store.name}")
for product in store.products:
    print(f"- {product.name}: ${product.price}")
```

## Data Extraction

### Extract Information from Text

```
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str
    company: str

text = """
Hi, I'm John Smith from TechCorp.
You can reach me at john.smith@techcorp.com or call 555-0123.
"""

response = client.structured_response(
    input=f"Extract contact information from this text: {text}",
    output_cls=ContactInfo
)

contact = response.structured_data[0]
print(f"Contact: {contact.name} at {contact.company}")
```

### Analyze and Categorize

```
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class TextAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)
    key_topics: List[str]
    summary: str

review = "This product is amazing! Great quality and fast shipping."

response = client.structured_response(
    input=f"Analyze this review: {review}",
    output_cls=TextAnalysis
)

analysis = response.structured_data[0]
print(f"Sentiment: {analysis.sentiment}")
print(f"Confidence: {analysis.confidence}")
print(f"Topics: {', '.join(analysis.key_topics)}")
```

---

<a id="4"></a>

## Streaming - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Clients/streaming/

# Streaming

Streaming allows you to receive responses in real-time as they're generated, providing a better user experience for long responses and interactive applications.

## Why Use Streaming?

* **Real-time feedback**: Users see responses as they're generated
* **Better UX**: Reduces perceived latency for long responses
* **Progressive display**: Show partial results immediately
* **Interruptible**: Can stop generation early if needed

## Basic Streaming

### Synchronous Streaming

```
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# Basic streaming
for chunk in client.stream_invoke("Write a short story about a robot learning to paint"):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
print()  # New line when complete
```

### Asynchronous Streaming

```
import asyncio

async def async_stream_example():
    async for chunk in client.a_stream_invoke("Explain quantum computing in simple terms"):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()  # New line when complete

# Run the async function
asyncio.run(async_stream_example())
```

---

<a id="5"></a>

## Tools - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Clients/tools/

# Tools

Tools allow AI models to call external functions, enabling them to perform actions, retrieve data, and interact with external systems.

## Basic Tool Usage

### Simple Tool

```
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

client = OpenAIClient(api_key="your-api-key", model="gpt-4o-mini")

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulate weather API call
    return f"The weather in {location} is sunny and 72°F"

# Use the tool
response = client.invoke(
    "What's the weather in New York?",
    tools=[get_weather]
)

# Execute tool calls
for func_call in response.function_calls:
    result = func_call.tool(**func_call.arguments)
    print(f"Tool result: {result}")

print(response.text)
```

### Multiple Tools

```
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)  # Note: Use safe evaluation in production
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Use multiple tools
response = client.invoke(
    "What time is it and what's 15 * 8?",
    tools=[get_time, calculate]
)

# Execute all tool calls
for func_call in response.function_calls:
    result = func_call.tool(**func_call.arguments)
    print(f"{func_call.name}: {result}")
```

## Tool Choice Control

### Auto (Default)

Let the model decide when to use tools:

```
response = client.invoke(
    "Hello, how are you?",
    tools=[get_weather],
    tool_choice="auto"  # Model may or may not use tools
)
```

### Required

Force the model to use a tool:

```
response = client.invoke(
    "Get weather information",
    tools=[get_weather],
    tool_choice="required"  # Model must use a tool
)
```

### None

Disable tool usage:

```
response = client.invoke(
    "What's the weather like?",
    tools=[get_weather],
    tool_choice="none"  # Model won't use tools
)
```

### Specific Tool

Force a specific tool:

```
response = client.invoke(
    "Check the weather",
    tools=[get_weather, get_time],
    tool_choice=["get_weather"]  # Only use this specific tool
)
```

---

<a id="6"></a>

## Real example: Chatbot - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Clients/chatbot/

# Real example: Chatbot

Learn how to build conversational AI applications using the OpenAI client with memory management, context awareness, and advanced chatbot patterns.

## Basic Chatbot

Clients need memory to maintain context and have meaningful conversations. The Memory class stores and manages conversation history, allowing the AI to reference previous exchanges and maintain coherent dialogue.

Here's a simple example of a chatbot with memory:

```
from datapizza.clients.openai import OpenAIClient
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant"
)

def simple_chatbot():
    """Basic chatbot with conversation memory."""

    memory = Memory()

    print("Chatbot: Hello! I'm here to help. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        # Get AI response with memory context
        response = client.invoke(user_input, memory=memory)
        print(f"Chatbot: {response.text}")

        # Update conversation memory
        memory.add_turn(TextBlock(content=user_input), role=ROLE.USER)
        memory.add_turn(response.content, role=ROLE.ASSISTANT)

# Run the chatbot
simple_chatbot()
```

---

<a id="7"></a>

## Running with Ollama - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Clients/local_model/

# Running with Ollama

Datapizza AI supports running with local models through Ollama, providing you with complete control over your AI infrastructure while maintaining privacy and reducing costs.

## Prerequisites

Before getting started, you'll need to have Ollama installed and running on your system.

### Installing Ollama

1. **Download and Install Ollama**
2. Visit [ollama.ai](https://ollama.ai) and download the installer for your operating system
3. Follow the installation instructions for your platform
4. **Start Ollama Service**

   ```
   # Ollama typically starts automatically after installation
   # If not, you can start it manually:
   ollama serve
   ```
5. **Pull a Model**

   ```
   # Pull the Gemma 2B model (lightweight option)
   ollama pull gemma2:2b

   # Or pull Gemma 7B for better performance
   ollama pull gemma2:7b

   # Or pull Llama 3.1 8B
   ollama pull llama3.1:8b
   ```

## Installation

Install the Datapizza AI OpenAI-like client:

```
pip install datapizza-ai-clients-openai-like
```

## Basic Usage

Here's a simple example of how to use Datapizza AI with Ollama:

```
import os
from datapizza.clients.openai_like import OpenAILikeClient
from dotenv import load_dotenv

load_dotenv()

# Create client for Ollama
client = OpenAILikeClient(
    api_key="",  # Ollama doesn't require an API key
    model="gemma2:2b",  # Use any model you've pulled with Ollama
    system_prompt="You are a helpful assistant.",
    base_url="http://localhost:11434/v1",  # Default Ollama API endpoint
)

# Simple query
response = client.invoke("What is the capital of France?")
print(response.content)
```

---


## Agents

<a id="8"></a>

## Build your first agent - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Agents/agent/

# Build your first agent

The `Agent` class is the core component for creating autonomous AI agents in Datapizza AI. It handles task execution, tool management, memory, and planning.

## Basic Usage

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.memory import Memory
from datapizza.tools import tool

agent = Agent(
    name="my_agent",
    system_prompt="You are a helpful assistant",
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    # tools=[],
    # max_steps=10,
    # terminate_on_text=True,  # Terminate execution when the client return a plain text
    # memory=memory,
    # stream=False,
    # planning_interval=0
)

res = agent.run("Hi")
print(res.text)
```

## Use Tools

The above agent is quite basic, so let's make it more functional by adding [**tools**](../../../API%20Reference/Type/tool/).

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.memory import Memory
from datapizza.tools import tool

@tool
def get_weather(location: str, when: str) -> str:
    """Retrieves weather information for a specified location and time."""
    return "25 °C"

agent = Agent(name="weather_agent", tools=[get_weather], client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"))
response = agent.run("What's the weather tomorrow in Milan?")

print(response.text)
# Output:
# Tomorrow in Milan, the temperature will be 25 °C.
```

### tool\_choice

You can set the parameter `tool_choice` at invoke time.

The accepted values are: `auto`, `required`, `none`, `required_first`, `list["tool_name"]`

```
res = master_agent.run(
    task_input="what is the weather in milan?", tool_choice="required_first"
)
```

* `auto`: the model will decide if use a tool or not.
* `required_first`: force to use a tool only at the first step, then auto.
* `required`: force to use a tool at every step.
* `none`: force to not use any tool.

## Core Methods

### Sync run

`run(task_input: str, tool_choice = "auto", **kwargs) -> str`
Execute a task and return the final result.

```
result = agent.run("What's the weather like today?")
print(result.text)  # "The weather is sunny with 25°C"
```

### Stream invoke

Stream the agent's execution process, yielding intermediate steps. (Do not stream the single answer)

```
from datapizza.agents.agent import Agent, StepResult
from datapizza.clients.openai import OpenAIClient
from datapizza.memory import Memory
from datapizza.tools import tool

@tool
def get_weather(location: str, when: str) -> str:
    """Retrieves weather information for a specified location and time."""
    return "25 °C"

agent = Agent(name="weather_agent", tools=[get_weather], client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"))

for step in agent.stream_invoke("What's the weather tomorrow in Milan?"):
    print(f"Step {step.index} starting...")
    print(step.text)
```

### Async run

`a_run(task_input: str, **kwargs) -> str`
Async version of run.

```
import asyncio

async def main():

    agent = Agent(name="agent", client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini"))
    return await agent.a_run("Process this request")


res = asyncio.run(main())
print(res.text)
```

### Async stream invoke

`a_stream_invoke(task_input: str, **kwargs) -> AsyncGenerator[str | StepResult, None]`
Stream the agent's execution process, yielding intermediate steps. (Do not stream the single answer)

```
from datapizza.agents.agent import Agent
from datapizza.clients.openai import OpenAIClient
import asyncio

async def get_response():
    client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4o-mini")
    agent = Agent(name= "joke_agent",client=client)
    async for step in agent.a_stream_invoke("tell me a joke"):
        print(f"Step {step.index} starting...")
        print(step.text)

asyncio.run(get_response())
```

## Multi-Agent Communication

An agent can call another ones using `can_call` method

```
from datapizza.agents.agent import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1")

@tool
def get_weather(city: str) -> str:
    return f""" Monday's weather in {city} is cloudy.
                Tuesday's weather in {city} is rainy.
                Wednesday's weather in {city} is sunny
                Thursday's weather in {city} is cloudy,
                Friday's weather in {city} is rainy,
                Saturday's weather in {city} is sunny
                and Sunday's weather in {city} is cloudy."""

weather_agent = Agent(
    name="weather_expert",
    client=client,
    system_prompt="You are a weather expert. Provide detailed weather information and forecasts.",
    tools=[get_weather]
)

planner_agent = Agent(
    name="planner",
    client=client,
    system_prompt="You are a trip planner. Use weather and analysis info to make recommendations."
)

planner_agent.can_call(weather_agent)

response = planner_agent.run(
    "I need to plan a hiking trip in Seattle next week. Can you help analyze the weather and make recommendations?"
)
print(response.text)
```

Alternatively, you can define a tool that manually calls the agent.
The two solutions are more or less identical.

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

class MasterAgent(Agent):
    system_prompt="You are a master agent. You can call the weather expert to get weather information."
    name="master_agent"

    @tool
    def call_weather_expert(self, task_to_ask: str) -> str:
        @tool
        def get_weather(city: str) -> str:
            return f""" Monday's weather in {city} is cloudy.
                        Tuesday's weather in {city} is rainy.
                        Wednesday's weather in {city} is sunny
                        Thursday's weather in {city} is cloudy,
                        Friday's weather in {city} is rainy,
                        Saturday's weather in {city} is sunny
                        and Sunday's weather in {city} is cloudy."""

        weather_agent = Agent(
            name="weather_expert",
            client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1"),
            system_prompt="You are a weather expert. Provide detailed weather information and forecasts.",
            tools=[get_weather]
        )
        res = weather_agent.run(task_to_ask)
        return res.text

master_agent = MasterAgent(
    client=OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1"),
)

master_agent.run("What is the weather in Rome?")
```

## Planning System

When `planning_interval > 0`, the agent creates execution plans at regular intervals:

During the planning stages, the agent spends time thinking about what the next steps are to be taken to achieve the task.

```
agent = Agent(
    name="Agent_with_plan",
    client=client,
    planning_interval=3,  # Plan every 3 steps
)
```

The planning system generates structured plans that help the agent organize complex tasks.

## Stream output response

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.core.clients import ClientResponse
from datapizza.tools import tool

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1")

agent = Agent(
    name="Big_boss",
    client=client,
    system_prompt="You are a helpful assistant that answers questions based on the provided context.",
    stream=True, # With stream=True, the agent will stream the client resposne, not only the intermediate steps

)

for r in agent.stream_invoke("What is the weather in Milan?"):
    if isinstance(r, ClientResponse):
        print(r.delta, end="", flush=True)
```

---

<a id="9"></a>

## Model Context Protocol (MCP) - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Agents/mcp/

# Model Context Protocol (MCP)

Model Context Protocol (MCP) is an open-source standard that enables AI applications to connect with external systems like databases, APIs, and tools.

Use MCP (Model Context Protocol) tools inside `datapizza-ai` by wrapping them as
regular agent tools. Follow this minimal recipe to get an agent talking to a
remote MCP server in just a few steps.

With MCP, you can build AI agents that:

* **Access your codebase**: Let AI read GitHub repositories, create issues, and manage pull requests
* **Query your database**: Enable natural language queries against PostgreSQL, MySQL, or any database
* **Browse the web**: Give AI the ability to search and extract information from websites
* **Control your tools**: Connect to Slack, Notion, Google Calendar, or any API-based service
* **Analyze your data**: Let AI work with spreadsheets, documents, and business intelligence tools

## Fetch MCP tools

Here an example of [FastMCP](https://gofastmcp.com/getting-started/welcome) tool provided by FastMCP

```
from datapizza.tools.mcp_client import MCPClient

fastmcp_client = MCPClient(url="https://gofastmcp.com/mcp")
fastmcp_tools = fastmcp_client.list_tools()
```

## Create the agent and run it

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o-mini")

agent = Agent(
    name="mcp_agent",
    client=client,
    tools=fastmcp_tools,
)

result = agent.run("How can I use a FastMCP server over HTTP?")
print(result.text)
```

That’s it—you now have an agent that discovers tools from the FastMCP server and
uses them as part of normal `datapizza-ai` reasoning. Swap in any MCP endpoint
or different LLM client to match your project.

---


## RAG

<a id="10"></a>

## Build a RAG - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/RAG/rag/

# Build a RAG

This guide demonstrates how to build a complete RAG (Retrieval-Augmented Generation) system using datapizza-ai's pipeline architecture. We'll cover both the **ingestion pipeline** for processing and storing documents, and the **DagPipeline** for retrieval and response generation.

## Overview

A RAG system consists of two main phases:

1. **Ingestion**: Process documents, split them into chunks, generate embeddings, and store in a vector database
2. **Retrieval**: Query the vector database, retrieve relevant chunks, and generate responses

datapizza-ai provides specialized pipeline components for each phase:

* **IngestionPipeline**: Sequential processing for document ingestion
* **DagPipeline**: Graph-based processing for complex retrieval workflows

## Part 1: Document Ingestion Pipeline

The ingestion pipeline processes raw documents and stores them in a vector database. Here's a complete example:

### Basic Ingestion Setup

```
pip install datapizza-ai-parsers-docling
```

```
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.captioners import LLMCaptioner
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

vectorstore = QdrantVectorstore(location=":memory:")
vectorstore.create_collection(
    "my_documents",
    vector_config=[VectorConfig(name="text-embedding-3-small", dimensions=1536)]
)

embedder_client = OpenAIEmbedder(
    api_key="YOUR_API_KEY",
    model_name="text-embedding-3-small",
)

ingestion_pipeline = IngestionPipeline(
    modules=[
        DoclingParser(), # choose between Docling, Azure or TextParser to parse plain text

        #LLMCaptioner(
        #    client=OpenAIClient(api_key="YOUR_API_KEY"),
        #), # This is optional, add it if you want to caption the media

        NodeSplitter(max_char=1000),             # Split Nodes into Chunks
        ChunkEmbedder(client=embedder_client),   # Add embeddings to Chunks
    ],
    vector_store=vectorstore,
    collection_name="my_documents"
)

ingestion_pipeline.run("sample.pdf", metadata={"source": "user_upload"})

res = vectorstore.search(
    query_vector = [0.0] * 1536,
    collection_name="my_documents",
    k=2,
)
print(res)
```

### Configuration-Based Ingestion

You can also define your pipeline using YAML configuration:

```
constants:
  EMBEDDING_MODEL: "text-embedding-3-small"
  CHUNK_SIZE: 1000

ingestion_pipeline:
  clients:
    openai_embedder:
      provider: openai
      model: "${EMBEDDING_MODEL}"
      api_key: "${OPENAI_API_KEY}"

  modules:
    - name: parser
      type: DoclingParser
      module: datapizza.modules.parsers.docling
    - name: splitter
      type: NodeSplitter
      module: datapizza.modules.splitters
      params:
        max_char: ${CHUNK_SIZE}
    - name: embedder
      type: ChunkEmbedder
      module: datapizza.embedders
      params:
        client: openai_embedder

  vector_store:
    type: QdrantVectorstore
    module: datapizza.vectorstores.qdrant
    params:
      host: "localhost"
      port: 6333

  collection_name: "my_documents"
```

Load and use the configuration:

```
from datapizza.pipeline import IngestionPipeline

# Make sure the collection exists before running the pipeline
pipeline = IngestionPipeline().from_yaml("ingestion_pipeline.yaml")
pipeline.run("sample.pdf")
```

## Part 2: Retrieval with DagPipeline

The DagPipeline enables complex retrieval workflows with query rewriting, embedding, and response generation.

### Basic Retrieval Setup

```
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

openai_client = OpenAIClient(
    model="gpt-4o-mini",
    api_key="YOUR_API_KEY"
)

query_rewriter = ToolRewriter(
    client=openai_client,
    system_prompt="Rewrite user queries to improve retrieval accuracy."
)

embedder = OpenAIEmbedder(
    api_key="YOUR_API_KEY",
    model_name="text-embedding-3-small"
)

# Use the same qdrant of ingestion (prefer host and port instead of location when possible)
retriever = QdrantVectorstore(location=":memory:")
retriever.create_collection(
    "my_documents",
    vector_config=[VectorConfig(name="embedding", dimensions=1536)]
)

prompt_template = ChatPromptTemplate(
    user_prompt_template="User question: {{user_prompt}}\n:",
    retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"
)

dag_pipeline = DagPipeline()
dag_pipeline.add_module("rewriter", query_rewriter)
dag_pipeline.add_module("embedder", embedder)
dag_pipeline.add_module("retriever", retriever)
dag_pipeline.add_module("prompt", prompt_template)
dag_pipeline.add_module("generator", openai_client)

dag_pipeline.connect("rewriter", "embedder", target_key="text")
dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
dag_pipeline.connect("retriever", "prompt", target_key="chunks")
dag_pipeline.connect("prompt", "generator", target_key="memory")

query = "tell me something about this document"
result = dag_pipeline.run({
    "rewriter": {"user_prompt": query},
    "prompt": {"user_prompt": query},
    "retriever": {"collection_name": "my_documents", "k": 3},
    "generator":{"input": query}
})

print(f"Generated response: {result['generator']}")
```

---


## Pipeline

<a id="11"></a>

## Ingestion Pipeline - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Pipeline/ingestion_pipeline/

# Ingestion Pipeline

The `IngestionPipeline` provides a streamlined way to process documents, transform them into nodes (chunks of text with metadata), generate embeddings, and optionally store them in a vector database. It allows chaining various components like parsers, captioners, splitters, and embedders to create a customizable document processing workflow.

## Core Concepts

* **Components**: These are the processing steps in the pipeline, typically inheriting from `datapizza.core.models.PipelineComponent`. Each component implements a `process` method to perform a specific task like parsing a document, splitting text, or generating embeddings. Components are executed sequentially via their `__call__` method in the order they are provided.
* **Vector Store**: An optional component responsible for storing the final nodes and their embeddings.
* **Nodes**: The fundamental unit of data passed between components. A node usually represents a chunk of text (e.g., a paragraph, a table summary) along with its associated metadata and embeddings.

## Available Components

The pipeline typically supports components for:

1. [**Parsers**](../../../API%20Reference/Modules/Parsers/): Convert raw documents (PDF, DOCX, etc.) into structured `Node` objects (e.g., `AzureParser`, `UnstructuredParser`).
2. [**Captioners**](../../../API%20Reference/Modules/captioners/): Enhance nodes representing images or tables with textual descriptions using models like LLMs (e.g., `LLMCaptioner`).
3. [**Splitters**](../../../API%20Reference/Modules/Splitters/): Divide nodes into smaller chunks based on their content (e.g., `NodeSplitter`, `PdfImageSplitter`).
4. [**Embedders**](../../../API%20Reference/Embedders/openai_embedder/): Create chunk embeddings for semantic search and similarity matching (e.g., `NodeEmbedder`, `ClientEmbedder`).
   * [`ChunkEmbedder`](../../../API%20Reference/Embedders/chunk_embedder/): Batch processing for efficient embedding of multiple nodes.
5. [**Vector Stores**](../../../API%20Reference/Vectorstore/qdrant_vectorstore/): Store and retrieve embeddings efficiently using vector databases (e.g., `QdrantVectorstore`).

Refer to the specific documentation for each component type (e.g., in `datapizza.parsers`, `datapizza.embedders`) for details on their specific parameters and usage. Remember that pipeline components typically inherit from `PipelineComponent` and implement the `_run` method.

## Configuration Methods

There are two main ways to configure and use the `IngestionPipeline`:

### 1. Programmatic Configuration

Define and configure the pipeline directly within your Python code. This offers maximum flexibility.

```
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

vector_store = QdrantVectorstore(
    location=":memory:" # or set host and port
)
vector_store.create_collection(collection_name="datapizza", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])

pipeline = IngestionPipeline(
    modules=[
        DoclingParser(),
        NodeSplitter(max_char=2000),
        ChunkEmbedder(client=OpenAIClient(api_key="OPENAI_API_KEY", model="text-embedding-3-small"), model_name="text-embedding-3-small", embedding_name="small"),
    ],
    vector_store=vector_store,
    collection_name="datapizza",
)

pipeline.run(file_path="sample.pdf")

print(vector_store.search(query_vector= [0.0]*1536, collection_name="datapizza", k=4))
```

### 2. YAML Configuration

Define the entire pipeline structure, components, and their parameters in a YAML file. This is useful for managing configurations separately from code.

```
from datapizza.pipeline.pipeline import IngestionPipeline
import os

# Load pipeline from YAML
pipeline = IngestionPipeline().from_yaml("path/to/your/config.yaml")

# Run the pipeline (Ensure necessary ENV VARS for the YAML config are set)
pipeline.run(file_path="path/to/your/document.pdf")
```

#### Example YAML Configuration (`config.yaml`)

```
constants:
  EMBEDDING_MODEL: "text-embedding-3-small"
  CHUNK_SIZE: 1000

ingestion_pipeline:
  clients:
    openai_embedder:
      provider: openai
      model: "${EMBEDDING_MODEL}"
      api_key: "${OPENAI_API_KEY}"

  modules:
    - name: parser
      type: DoclingParser
      module: datapizza.modules.parsers.docling
    - name: splitter
      type: NodeSplitter
      module: datapizza.modules.splitters
      params:
        max_char: ${CHUNK_SIZE}
    - name: embedder
      type: ChunkEmbedder
      module: datapizza.embedders
      params:
        client: openai_embedder

  vector_store:
    type: QdrantVectorstore
    module: datapizza.vectorstores.qdrant
    params:
      host: "localhost"
      port: 6333

  collection_name: "my_documents"
```

**Key points for YAML configuration:**

* **Environment Variables**: Use `${VAR_NAME}` syntax within strings to securely load secrets or configuration from environment variables. Ensure these variables are set in your execution environment.
* **Clients**: Define shared clients (like `OpenAIClient`) under the `clients` key and reference them by name within module `params`.
* **Modules**: List components under `modules`. Each requires `type` (class name) and `module` (Python path to the class). `params` are passed to the component's constructor (`__init__`). Components should generally inherit from `PipelineComponent`.
* **Vector Store**: Configure the optional vector store similarly to modules.
* **Collection Name**: Must be provided if a `vector_store` is configured.

## Pipeline Execution (`run` method)

```
pipeline.run(file_path=f, metadata={"name": f, "type": "md"})
```

### Async Execution (`a_run` method)

IngestionPipeline support async run
*NB:* Every modules should implement `_a_run` method to run the async pipeline.

```
await pipeline.a_run(file_path=f, metadata={"name": f, "type": "md"})
```

---

<a id="12"></a>

## DagPipeline - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Pipeline/retrieval_pipeline/

# DagPipeline

The `DagPipeline` class allows you to define and execute a series of processing steps (modules) organized as a Directed Acyclic Graph (DAG). Modules typically inherit from `datapizza.core.models.PipelineComponent` or are simple callables. This enables complex workflows where the output of one module can be selectively used as input for others.

## Core Concepts

### Modules

Modules are the building blocks of the pipeline. They are typically instances of classes inheriting from `datapizza.core.models.PipelineComponent` (which requires implementing a `run` and `a_run` method), `datapizza.core.models.ChainableProducer` (which exposes an `as_module_component` method returning a `PipelineComponent`), or simply Python callables.

```
from datapizza.core.models import PipelineComponent
from datapizza.pipeline import DagPipeline

class MyProcessingStep(PipelineComponent):
    # Inheriting from PipelineComponent provides the __call__ wrapper for logging
    def _run(self, input_data: str) -> str:
        return something

    async _a_run(self, something: str) -> str:
        return await do_stuff()
```

### Connections

Connections define the flow of data between modules. You specify which module's output connects to which module's input.

* **`from_node_name`**: The name of the source module.
* **`to_node_name`**: The name of the target module.
* **`source_key`** (Optional): If the source module's `process` method (or callable) returns a dictionary, this key specifies which value from the dictionary should be passed. If `None`, the entire output of the source module is passed.
* **`target_key`** : This key specifies the argument name in the target module's `process` method (or callable) that should receive the data. If `None`, and the source output is *not* a dictionary, the data is passed as the first non-`self` argument to the target's `_run` method/callable. If `None` and the source output *is* a dictionary, its key-value pairs are merged into the target's input keyword arguments.

```
from datapizza.clients.openai import OpenAIClient
from datapizza.core.models import PipelineComponent
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o-mini")
vector_store = QdrantVectorstore(location=":memory:")
vector_store.create_collection(collection_name="my_documents", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])

pipeline = DagPipeline()

pipeline.add_module("rewriter", ToolRewriter(client=client, system_prompt="rewrite the query to perform a better search in a vector database"))
pipeline.add_module("embedder", OpenAIEmbedder(api_key="OPENAI_API_KEY", model_name="text-embedding-3-small"))
pipeline.add_module("vector_store", vector_store)
pipeline.add_module("prompt_template", ChatPromptTemplate(user_prompt_template = "this is a user prompt: {{ user_prompt }}", retrieval_prompt_template = "{% for chunk in chunks %} Relevant chunk: {{ chunk.text }} \n\n {% endfor %}"))
pipeline.add_module("llm", OpenAIClient(model = "gpt-4o-mini", api_key = "OPENAI_API_KEY"))


pipeline.connect("rewriter", "embedder", target_key="text")
pipeline.connect("embedder", "vector_store", target_key="query_vector")
pipeline.connect("vector_store", "prompt_template", target_key="chunks")
pipeline.connect("prompt_template", "llm", target_key="memory")
```

## Running the Pipeline

The `run` method executes the pipeline based on the defined connections. It requires an initial `data` dictionary which provides the missing input arguments for the nodes that require them.

The keys of this dictionary should match the names of the modules requiring initial input, and the values should be dictionaries mapping argument names to values for their respective `process` methods (or callables).

```
user_input = "tell me something about this document"
res = pipeline.run(
    {
        "rewriter": {"user_prompt": user_input},

        # Embedder doesn't require any input because it's provided by the rewriter

        "prompt_template": {"user_prompt": user_input},  # Prompt template requires user_prompt
        "vector_store": {
            "collection_name": "my_documents",
            "k": 10,
        },
        "llm": {
            "input": user_input,
            "system_prompt": "You are a helpful assistant. try to answer user questions given the context",
        },
    }
)
result = res.get("llm").text
print(result)
```

The pipeline automatically determines the execution order based on dependencies. It executes modules by calling their `run` method only when all their prerequisites (connected `from_node_name` modules) have completed successfully.

### Async run

Pipeline support async run with `a_run`
With async run, the pipeline will call a\_run of modules.

This only works if you are using a remote qdrant server. The in-memory qdrant function does not work with asynchronous execution.

```
res = await pipeline.a_run(
    {
        "rewriter": {"user_prompt": user_input},
        "prompt_template": {"user_prompt": user_input},
        "vector_store": {
            "collection_name": "datapizza",
            "k": 10,
        },
        "llm": {
            "input": user_input,
            "system_prompt": "You are a helpful assistant. try to answer user questions given the context",
        },
    }
)
```

## Configuration via YAML

Pipelines can be defined entirely using a YAML configuration file, which is loaded using the `from_yaml` method. This is useful for separating pipeline structure from code.

The YAML structure includes sections for `clients` (like LLM providers), `modules`, and `connections`.

```
from datapizza.pipeline import DagPipeline

pipeline = DagPipeline().from_yaml("dag_pipeline.yaml")
user_input = "tell me something about this document"
res = pipeline.run(
    {
        "rewriter": {"user_prompt": user_input},
        "prompt_template": {"user_prompt": user_input},
        "vector_store": {"collection_name": "my_documents","k": 10,},
        "llm": {"input": user_input,"system_prompt": "You are a helpful assistant. try to answer user questions given the context",},
    }
)
result = res.get("llm").text
print(result)
```

### Example YAML (`dag_config.yaml`)

```
dag_pipeline:
  clients:
    openai_client:
      provider: openai
      model: "gpt-4o-mini"
      api_key: ${OPENAI_API_KEY}
    google_client:
      provider: google
      model: "gemini-2.0"
      api_key: ${GOOGLE_API_KEY}
    openai_embedder:
      provider: openai
      model: "text-embedding-3-small"
      api_key: ${OPENAI_API_KEY}

  modules:
    - name: rewriter
      type: ToolRewriter
      module: datapizza.modules.rewriters
      params:
        client: openai_client
        system_prompt: "rewrite the query to perform a better search in a vector database"
    - name: embedder
      type: ClientEmbedder
      module: datapizza.embedders
      params:
        client: openai_embedder
    - name: vector_store
      type: QdrantVectorstore
      module: datapizza.vectorstores.qdrant
      params:
        host: localhost
    - name: prompt_template
      type: ChatPromptTemplate
      module: datapizza.modules.prompt
      params:
        user_prompt_template: "this is a user prompt: {{ user_prompt }}"
        retrieval_prompt_template: "{% for chunk in chunks %} Relevant chunk: {{ chunk.text }} \n\n {% endfor %}"
    - name: llm
      type: OpenAIClient
      module: datapizza.clients.openai
      params:
        model: "gpt-4o-mini"
        api_key: ${OPENAI_API_KEY}

  connections:

    - from: rewriter
      to: embedder
      target_key: text
    - from: embedder
      to: vector_store
      target_key: query_vector
    - from: vector_store
      to: prompt_template
      target_key: chunks
    - from: prompt_template
      to: llm
      target_key: memory
```

**Key points for YAML configuration:**

* **Environment Variables**: Use `${VAR_NAME}` syntax to load sensitive information like API keys from environment variables.
* **Clients**: Define clients once and reference them by name in module `params`.
* **Module Loading**: Specify the `module` path and `type` (class name) for dynamic loading. The class should generally be a `PipelineComponent`.
* **Parameters**: `params` are passed directly to the module's constructor.
* **Connections**: Define data flow similarly to the programmatic `connect` method.

---

<a id="13"></a>

## Functional Pipeline - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Pipeline/functional_pipeline/

# Functional Pipeline

> ***WARNING:*** This module is in beta. Signatures and interfaces may change in future releases.

The `FunctionalPipeline` module provides a flexible way to build data processing pipelines with complex dependency graphs. It allows you to define reusable processing nodes and connect them in various patterns including sequential execution, branching, parallel execution, and foreach loops.

## Core Components

### Dependency

Defines how data flows between [Nodes](../../../API%20Reference/Type/node/):

```
@dataclass
class Dependency:
    node_name: str
    input_key: str | None = None
    target_key: str | None = None
```

* `node_name`: The name of the node to get data from
* `input_key`: Optional key for extracting a specific part of the node's output
* `target_key`: The key under which to store the data in the receiving node's input

### FunctionalPipeline

The main class for building and executing pipelines:

```
class FunctionalPipeline:
    def __init__(self):
        self.nodes = []
```

## Building Pipelines

### Sequential Execution

```
pipeline = FunctionalPipeline()
pipeline.run("load_data", DataLoader(), kwargs={"filepath": "data.csv"})
pipeline.then("transform", Transformer(), target_key="data")
pipeline.then("save", Saver(), target_key="transformed_data")
```

### Branching

```
pipeline.branch(
    condition=is_valid_data,
    if_true=valid_data_pipeline,
    if_false=invalid_data_pipeline,
    dependencies=[Dependency(node_name="validate", target_key="validation_result")]
)
```

### Foreach Loop

```
pipeline.foreach(
    name="process_items",
    do=item_processing_pipeline,
    dependencies=[Dependency(node_name="get_items")]
)
```

## Executing Pipelines

```
result = pipeline.execute(
    initial_data={"load_data": {"filepath": "override.csv"}},
    context={"existing_data": {...}}
)
```

## YAML Configuration

You can define pipelines in YAML and load them at runtime:
This is useful for separating pipeline structure from code

```
modules:
  - name: data_loader
    module: my_package.loaders
    type: CSVLoader
    params:
      encoding: "utf-8"

  - name: transformer
    module: my_package.transformers
    type: StandardTransformer

pipeline:
  - type: run
    name: load_data
    node: data_loader
    kwargs:
      filepath: "data.csv"

  - type: then
    name: transform
    node: transformer
    target_key: data
```

Load the pipeline:

```
pipeline = FunctionalPipeline.from_yaml("pipeline_config.yaml")
result = pipeline.execute()
```

## Real-world Examples

### Question Answering Pipeline

Here's an example of a question answering pipeline that uses embeddings to retrieve relevant information and an LLM to generate a response:

Define the components:

```
from datapizza.clients.google import GoogleClient
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import Dependency, FunctionalPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from dotenv import load_dotenv

load_dotenv()

rewriter = ToolRewriter(
    client=OpenAIClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt="Use only 1 time the tool to answer the user prompt.",
    )
)
embedder = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
)

vector_store = QdrantVectorstore(host="localhost", port=6333)
vector_store.create_collection(collection_name="my_documents", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])
vector_store = vector_store.as_module_component() # required to use the vectorstore in the pipeline

prompt_template = ChatPromptTemplate(
    user_prompt_template="this is a user prompt: {{ user_prompt }}",
    retrieval_prompt_template="{% for chunk in chunks %} Relevant chunk: {{ chunk.text }} \n\n {% endfor %}",
)
generator = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    system_prompt="You are a senior Software Engineer. You are given a user prompt and you need to answer it given the context of the chunks.",
).as_module_component()
```

And now create and execute the pipeline

```
pipeline = (FunctionalPipeline()
    .run(name="rewriter", node=rewriter, kwargs={"user_prompt": "tell me something about this document"})
    .then(name="embedder", node=embedder, target_key="text")
    .then(name="vector_store", node=vector_store, target_key="query_vector",
          kwargs={"collection_name": "my_documents", "k": 4})
    .then(name="prompt_template", node=prompt_template, target_key="chunks" , kwargs={"user_prompt": "tell me something about this document"})
    .then(name="generator", node=generator, target_key="memory", kwargs={"input": "tell me something about this document"})
    .get("generator")
)

result = pipeline.execute()
print(result)
```

When using `.then()`, the `target_key` parameter specifies the input parameter name for the current node's `run()` method that will receive the output from the previous node. In other words, `target_key` defines how the previous node's output gets mapped into the current node's `run()` method parameters.

This pipeline:

1. [Rewrites/processes](../../../API%20Reference/Modules/rewriters/) the user query
2. [Creates embeddings](../../../API%20Reference/Embedders/chunk_embedder/) from the processed query
3. Retrieves relevant chunks from a [vector database](../../../API%20Reference/Vectorstore/qdrant_vectorstore/)
4. [Creates a prompt template](../../../API%20Reference/Modules/Prompt/ChatPromptTemplate/) with the retrieved context
5. Generates a response using an LLM
6. Returns the generated response

### Branch and loop usage example

```
from datapizza.core.models import PipelineComponent
from datapizza.pipeline import Dependency, FunctionalPipeline


class Scraper(PipelineComponent):
    def _run(self, number_of_links: int = 1):
        return ["example.com"] * number_of_links

class UpperComponent(PipelineComponent):
    def _run(self, item):
        return item.upper()

class SendNotification(PipelineComponent):
    def _run(self ):
        return "No Url found, Notification sent"

send_notification = FunctionalPipeline().run(name="send_notification", node=SendNotification())

upper_elements = FunctionalPipeline().foreach(
    name="loop_links",
    dependencies=[Dependency(node_name="get_link")],
    do=UpperComponent(),
)

pipeline = (
    FunctionalPipeline()
    .run(name="get_link", node=Scraper())
    .branch(
        condition=lambda pipeline_context: len(pipeline_context.get("get_link")) > 0,
        dependencies=[Dependency(node_name="get_link")],
        if_true=upper_elements,
        if_false=send_notification,
    )
)

results = pipeline.execute(initial_data={"get_link": {"number_of_links": 0}}) # put 1 to test the other branch
print(results)
```

---


## Monitoring

<a id="14"></a>

## Tracing - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Monitoring/tracing/

# Tracing

The tracing module provides an easy-to-use interface for collecting and displaying OpenTelemetry traces with rich console output. It's designed to help developers monitor performance and understand the execution flow of their applications.

## Features

* **In-memory trace collection** - Stores spans in memory for fast access
* **Context-aware tracking** - Only collects spans for explicitly tracked operations
* **Thread-safe operations** - Safe for use in multi-threaded applications
* **OpenTelemetry integration** - Works with standard OpenTelemetry instrumentation

## Quick Start

The simplest way to use tracing is with the `tracer` context manager:

```
from datapizza.tracing import ContextTracing


# Basic tracing
with ContextTracing().trace("trace_name"):
    # Your code here
    result = some_datapizza_operations()

# Output will show:
# ╭─ Trace Summary of my_operation ────────────────────────────────── ╮
# │ Total Spans: 3                                                    │
# │ Duration: 2.45s                                                   │
# │ ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
# │ ┃ Model       ┃ Prompt Tokens ┃ Completion Tokens ┃ Cached Tokens ┃
# │ ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
# │ │ gpt-4o-mini │ 31            │ 27                │ 0             │
# │ └─────────────┴───────────────┴───────────────────┴───────────────┘
# ╰───────────────────────────────────────────────────────────────────╯
```

## Clients trace input/output/memory

If you want to log the input/output and the memory passed to client invoke you should set the env variable

`DATAPIZZA_TRACE_CLIENT_IO=TRUE`

default is `FALSE`

## Manual Span Creation

For more granular control, create spans manually:

```
from opentelemetry import trace
from datapizza.tracing import ContextTracing

tracer = trace.get_tracer(__name__)

with ContextTracing().trace("trace_name"):
    with tracer.start_as_current_span("database_query"):
        # Database operation
        data = fetch_from_database()

    with tracer.start_as_current_span("data_validation"):
        # Validation logic
        validate_data(data)

    with tracer.start_as_current_span("business_logic"):
        # Core business logic
        result = process_business_rules(data)
```

## Adding External Exporters

The tracing module uses in-memory storage by default, but you can easily add external exporters to send traces to other systems.

### Create the resource

First of all you should set the trace provider

```
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

resource = Resource.create(
   {
       ResourceAttributes.SERVICE_NAME: "your_service_name",
   }
)
trace.set_tracer_provider(TracerProvider(resource=resource))
```

### Zipkin Integration

Export traces to Zipkin for visualization and analysis:

`pip install opentelemetry-exporter-zipkin`

After setting the trace provider you can add the exporters

```
from opentelemetry import trace
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

zipkin_url = "http://localhost:9411/api/v2/spans"

zipkin_exporter = ZipkinExporter(
    endpoint=zipkin_url,
)

tracer_provider = trace.get_tracer_provider()

span_processor = SimpleSpanProcessor(zipkin_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Now all traces will be sent to both in-memory storage and Zipkin
```

### OTLP (OpenTelemetry Protocol)

Export to any OTLP-compatible backend (Grafana, Datadog, etc.):

```
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    headers={"authorization": "Bearer your-token"}
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

## Performance Considerations

* Use `BatchSpanProcessor` for external exporters in production
* Set reasonable limits on span attributes and events
* Monitor memory usage with many active traces

```
# Production configuration
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Batch spans for better performance
batch_processor = BatchSpanProcessor(
    exporter,
    max_queue_size=2048,
    schedule_delay_millis=5000,
    max_export_batch_size=512,
)

trace.get_tracer_provider().add_span_processor(batch_processor)
```

---

<a id="15"></a>

## Log level - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/Guides/Monitoring/log/

# Log level

With the variables `DATAPIZZA_LOG_LEVEL` and `DATAPIZZA_AGENT_LOG_LEVEL` you can change the log levels of the master logger and the agent logger

Allowed values are:

* `DEBUG`
* `INFO`
* `WARN`
* `ERROR`

The default values are:

* `DATAPIZZA_LOG_LEVEL=INFO`
* `DATAPIZZA_AGENT_LOG_LEVEL=INFO`

---


# 🛠️ API REFERENCE


## Clients

<a id="17"></a>

## Clients - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/clients/

# Clients

## datapizza.core.clients.client.Client

Bases: `ChainableProducer`

Represents the base class for all clients.
Concrete implementations must implement the abstract methods to handle the actual inference.

### a\_embed `async`

```
a_embed(text, model_name=None, **kwargs)
```

Embed a text using the model

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` | `str | list[str]` | The text to embed | *required* |
| `model_name` | `str` | The name of the model to use. Defaults to None. | `None` |
| `**kwargs` |  | Additional keyword arguments to pass to the model's embedding method | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `list[float] | list[list[float]]` | list[float]: The embedding vector for the text |

### a\_invoke `async`

```
a_invoke(
    input,
    tools=None,
    memory=None,
    tool_choice="auto",
    temperature=None,
    max_tokens=None,
    system_prompt=None,
    **kwargs,
)
```

Performs a single inference request to the model.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input` | `str` | The input text/prompt to send to the model | *required* |
| `tools` | `List[Tool]` | List of tools available for the model to use. Defaults to []. | `None` |
| `memory` | `Memory` | Memory object containing conversation history. Defaults to None. | `None` |
| `tool_choice` | `str` | Controls which tool to use. Defaults to "auto". | `'auto'` |
| `temperature` | `float` | Controls randomness in responses. Defaults to None. | `None` |
| `max_tokens` | `int` | Maximum number of tokens in the response. Defaults to None. | `None` |
| `system_prompt` | `str` | System-level instructions for the model. Defaults to None. | `None` |
| `**kwargs` |  | Additional keyword arguments to pass to the model's inference method | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `ClientResponse` | A ClientResponse object containing the model's response |

### a\_stream\_invoke `async`

```
a_stream_invoke(
    input,
    tools=None,
    memory=None,
    tool_choice="auto",
    temperature=None,
    max_tokens=None,
    system_prompt=None,
    **kwargs,
)
```

Streams the model's response token by token asynchronously.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input` | `str` | The input text/prompt to send to the model | *required* |
| `tools` | `List[Tool]` | List of tools available for the model to use. Defaults to []. | `None` |
| `memory` | `Memory` | Memory object containing conversation history. Defaults to None. | `None` |
| `tool_choice` | `str` | Controls which tool to use. Defaults to "auto". | `'auto'` |
| `temperature` | `float` | Controls randomness in responses. Defaults to None. | `None` |
| `max_tokens` | `int` | Maximum number of tokens in the response. Defaults to None. | `None` |
| `system_prompt` | `str` | System-level instructions for the model. Defaults to None. | `None` |
| `**kwargs` |  | Additional keyword arguments to pass to the model's inference method | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `AsyncIterator[ClientResponse]` | An async iterator yielding ClientResponse objects containing the model's response |

### a\_structured\_response `async`

```
a_structured_response(
    *,
    input,
    output_cls,
    memory=None,
    temperature=None,
    max_tokens=None,
    system_prompt=None,
    tools=None,
    tool_choice="auto",
    **kwargs,
)
```

Structures the model's response according to a specified output class.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input` | `str` | The input text/prompt to send to the model | *required* |
| `output_cls` | `Type[Model]` | The class type to structure the response into | *required* |
| `memory` | `Memory` | Memory object containing conversation history. Defaults to None. | `None` |
| `temperature` | `float` | Controls randomness in responses. Defaults to None. | `None` |
| `max_tokens` | `int` | Maximum number of tokens in the response. Defaults to None. | `None` |
| `system_prompt` | `str` | System-level instructions for the model. Defaults to None. | `None` |
| `tools` | `List[Tool]` | List of tools available for the model to use. Defaults to []. | `None` |
| `tool_choice` | `Literal['auto', 'required', 'none'] | list[str]` | Controls which tool to use ("auto" by default). Defaults to "auto". | `'auto'` |
| `**kwargs` |  | Additional keyword arguments to pass to the model's inference method | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `ClientResponse` | A ClientResponse object containing the structured response |

### embed

```
embed(text, model_name=None, **kwargs)
```

Embed a text using the model

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` | `str | list[str]` | The text to embed | *required* |
| `model_name` | `str` | The name of the model to use. Defaults to None. | `None` |
| `**kwargs` |  | Additional keyword arguments to pass to the model's embedding method | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `list[float]` | list[float]: The embedding vector for the text |

### invoke

```
invoke(
    input,
    tools=None,
    memory=None,
    tool_choice="auto",
    temperature=None,
    max_tokens=None,
    system_prompt=None,
    **kwargs,
)
```

Performs a single inference request to the model.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input` | `str` | The input text/prompt to send to the model | *required* |
| `tools` | `List[Tool]` | List of tools available for the model to use. Defaults to []. | `None` |
| `memory` | `Memory` | Memory object containing conversation history. Defaults to None. | `None` |
| `tool_choice` | `str` | Controls which tool to use. Defaults to "auto". | `'auto'` |
| `temperature` | `float` | Controls randomness in responses. Defaults to None. | `None` |
| `max_tokens` | `int` | Maximum number of tokens in the response. Defaults to None. | `None` |
| `system_prompt` | `str` | System-level instructions for the model. Defaults to None. | `None` |
| `**kwargs` |  | Additional keyword arguments to pass to the model's inference method | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `ClientResponse` | A ClientResponse object containing the model's response |

### stream\_invoke

```
stream_invoke(
    input,
    tools=None,
    memory=None,
    tool_choice="auto",
    temperature=None,
    max_tokens=None,
    system_prompt=None,
    **kwargs,
)
```

Streams the model's response token by token.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input` | `str` | The input text/prompt to send to the model | *required* |
| `tools` | `List[Tool]` | List of tools available for the model to use. Defaults to []. | `None` |
| `memory` | `Memory` | Memory object containing conversation history. Defaults to None. | `None` |
| `tool_choice` | `str` | Controls which tool to use. Defaults to "auto". | `'auto'` |
| `temperature` | `float` | Controls randomness in responses. Defaults to None. | `None` |
| `max_tokens` | `int` | Maximum number of tokens in the response. Defaults to None. | `None` |
| `system_prompt` | `str` | System-level instructions for the model. Defaults to None. | `None` |
| `**kwargs` |  | Additional keyword arguments to pass to the model's inference method | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `Iterator[ClientResponse]` | An iterator yielding ClientResponse objects containing the model's response |

### structured\_response

```
structured_response(
    *,
    input,
    output_cls,
    memory=None,
    temperature=None,
    max_tokens=None,
    system_prompt=None,
    tools=None,
    tool_choice="auto",
    **kwargs,
)
```

Structures the model's response according to a specified output class.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input` | `str` | The input text/prompt to send to the model | *required* |
| `output_cls` | `Type[Model]` | The class type to structure the response into | *required* |
| `memory` | `Memory` | Memory object containing conversation history. Defaults to None. | `None` |
| `temperature` | `float` | Controls randomness in responses. Defaults to None. | `None` |
| `max_tokens` | `int` | Maximum number of tokens in the response. Defaults to None. | `None` |
| `system_prompt` | `str` | System-level instructions for the model. Defaults to None. | `None` |
| `tools` | `List[Tool]` | List of tools available for the model to use. Defaults to []. | `None` |
| `tool_choice` | `Literal['auto', 'required', 'none'] | list[str]` | Controls which tool to use ("auto" by default). Defaults to "auto". | `'auto'` |
| `**kwargs` |  | Additional keyword arguments to pass to the model's inference method | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `ClientResponse` | A ClientResponse object containing the structured response |

---

<a id="18"></a>

## Client Factory - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/client_factory/

# Client Factory

The ClientFactory provides a convenient way to create LLM clients for different providers without having to import and instantiate each client type individually.

## datapizza.clients.factory.ClientFactory

Factory for creating LLM clients

### create `staticmethod`

```
create(
    provider,
    api_key,
    model,
    system_prompt="",
    temperature=0.7,
    **kwargs,
)
```

Create a client instance based on the specified provider.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `provider` | `str | Provider` | The LLM provider to use (openai, google, or anthropic) | *required* |
| `api_key` | `str` | API key for the provider | *required* |
| `model` | `str` | Model name to use (provider-specific) | *required* |
| `system_prompt` | `str` | System prompt to use | `''` |
| `temperature` | `float` | Temperature for generation (0-2) | `0.7` |
| `**kwargs` |  | Additional provider-specific arguments | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `Client` | An instance of the appropriate client |

Raises:

| Type | Description |
| --- | --- |
| `ValueError` | If the provider is not supported |

## Example Usage

```
from datapizza.clients.factory import ClientFactory, Provider

# Create an OpenAI client
openai_client = ClientFactory.create(
    provider=Provider.OPENAI,
    api_key="OPENAI_API_KEY",
    model="gpt-4",
    system_prompt="You are a helpful assistant.",
    temperature=0.7
)

# Create a Google client using string provider
google_client = ClientFactory.create(
    provider="google",
    api_key="GOOGLE_API_KEY",
    model="gemini-pro",
    system_prompt="You are a helpful assistant.",
    temperature=0.5
)

# Create an Anthropic client with custom parameters
anthropic_client = ClientFactory.create(
    provider=Provider.ANTHROPIC,
    api_key="ANTHROPIC_API_KEY",
    model="claude-3-sonnet-20240229",
    system_prompt="You are a helpful assistant.",
    temperature=0.3,
)

# Use the client
response = openai_client.invoke("What is the capital of France?")
print(response.content)
```

## Supported Providers

* `openai` - OpenAI GPT models
* `google` - Google Gemini models
* `anthropic` - Anthropic Claude models
* `mistral` - Mistral AI models

---

<a id="24"></a>

## Response - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/models/

# Response

## datapizza.core.clients.ClientResponse

A class for storing the response from a client.
Contains a list of blocks that can be text, function calls, or structured data,
maintaining the order in which they were generated.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `content` | `List[Block]` | A list of blocks. | *required* |
| `delta` | `str` | The delta of the response. Used for streaming responses. | `None` |
| `usage` | `TokenUsage` | Aggregated token usage. | `None` |
| `stop_reason` | `str` | Stop reason. | `None` |

### first\_text `property`

```
first_text
```

Returns the content of the first TextBlock or None

### function\_calls `property`

```
function_calls
```

Returns all function calls in order

### structured\_data `property`

```
structured_data
```

Returns all structured data in order

### text `property`

```
text
```

Returns concatenated text from all TextBlocks in order

### thoughts `property`

```
thoughts
```

Returns all thoughts in order

### is\_pure\_function\_call

```
is_pure_function_call()
```

Returns True if response contains only FunctionCallBlocks

### is\_pure\_text

```
is_pure_text()
```

Returns True if response contains only TextBlocks

---

<a id="25"></a>

## Cache - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/cache/

# Cache

## datapizza.core.cache.cache.Cache

Bases: `ABC`

This is the abstract base class for all cache implementations.
Concrete subclasses must provide implementations for the abstract methods that define how caching is handled.

When a cache instance is attached to a client, it will automatically store the results of the client`s method calls.
If the same method is invoked multiple times with identical arguments, the cache returns the stored result instead of re-executing the method.

### get `abstractmethod`

```
get(key)
```

Retrieve an object from the cache.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `key` | `str` | The key to retrieve the object for. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `object` | The object stored in the cache. |

### set `abstractmethod`

```
set(key, value)
```

Store an object in the cache.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `key` | `str` | The key to store the object for. | *required* |
| `value` | `str` | The object to store in the cache. | *required* |

---


### Available Clients

<a id="19"></a>

## Openai - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/Avaiable_Clients/openai/

# Openai

```
pip install datapizza-ai-clients-openai
```

## datapizza.clients.openai.OpenAIClient

Bases: `Client`

A client for interacting with the OpenAI API.

This class provides methods for invoking the OpenAI API to generate responses
based on given input data. It extends the Client class.

### \_\_init\_\_

```
__init__(
    api_key,
    model="gpt-4o-mini",
    system_prompt="",
    temperature=None,
    cache=None,
    base_url=None,
    organization=None,
    project=None,
    webhook_secret=None,
    websocket_base_url=None,
    timeout=None,
    max_retries=2,
    default_headers=None,
    default_query=None,
    http_client=None,
)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str` | The API key for the OpenAI API. | *required* |
| `model` | `str` | The model to use for the OpenAI API. | `'gpt-4o-mini'` |
| `system_prompt` | `str` | The system prompt to use for the OpenAI API. | `''` |
| `temperature` | `float | None` | The temperature to use for the OpenAI API. | `None` |
| `cache` | `Cache | None` | The cache to use for the OpenAI API. | `None` |
| `base_url` | `str | URL | None` | The base URL for the OpenAI API. | `None` |
| `organization` | `str | None` | The organization ID for the OpenAI API. | `None` |
| `project` | `str | None` | The project ID for the OpenAI API. | `None` |
| `webhook_secret` | `str | None` | The webhook secret for the OpenAI API. | `None` |
| `websocket_base_url` | `str | URL | None` | The websocket base URL for the OpenAI API. | `None` |
| `timeout` | `float | Timeout | None` | The timeout for the OpenAI API. | `None` |
| `max_retries` | `int` | The max retries for the OpenAI API. | `2` |
| `default_headers` | `dict[str, str] | None` | The default headers for the OpenAI API. | `None` |
| `default_query` | `dict[str, object] | None` | The default query for the OpenAI API. | `None` |
| `http_client` | `Client | None` | The http\_client for the OpenAI API. | `None` |

## Usage example

```
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(
    api_key="YOUR_API_KEY",
    model="gpt-4o-mini",
)
response = client.invoke("Hello!")
print(response.text)
```

## Include thinking

```
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(
    model= "gpt-5",
    api_key="YOUR_API_KEY",
)

response = client.invoke("Hi",reasoning={
        "effort": "low",
        "summary": "auto"
    }
)
print(response)
```

---

<a id="21"></a>

## Google - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/Avaiable_Clients/google/

# Google

```
pip install datapizza-ai-clients-google
```

## datapizza.clients.google.GoogleClient

Bases: `Client`

A client for interacting with Google's Generative AI APIs.

This class provides methods for invoking the Google GenAI API to generate responses
based on given input data. It extends the Client class.

### \_\_init\_\_

```
__init__(
    api_key=None,
    model="gemini-2.0-flash",
    system_prompt="",
    temperature=None,
    cache=None,
    project_id=None,
    location=None,
    credentials_path=None,
    use_vertexai=False,
)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str | None` | The API key for the Google API. | `None` |
| `model` | `str` | The model to use for the Google API. | `'gemini-2.0-flash'` |
| `system_prompt` | `str` | The system prompt to use for the Google API. | `''` |
| `temperature` | `float | None` | The temperature to use for the Google API. | `None` |
| `cache` | `Cache | None` | The cache to use for the Google API. | `None` |
| `project_id` | `str | None` | The project ID for the Google API. | `None` |
| `location` | `str | None` | The location for the Google API. | `None` |
| `credentials_path` | `str | None` | The path to the credentials for the Google API. | `None` |
| `use_vertexai` | `bool` | Whether to use Vertex AI for the Google API. | `False` |

# Usage example

```
import os

from datapizza.clients.google import GoogleClient
from dotenv import load_dotenv

load_dotenv()

client = GoogleClient(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.invoke("Hello!")
print(response.text)
```

---

<a id="20"></a>

## Anthropic - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/Avaiable_Clients/anthropic/

# Anthropic

```
pip install datapizza-ai-clients-anthropic
```

## datapizza.clients.anthropic.AnthropicClient

Bases: `Client`

A client for interacting with the Anthropic API (Claude).

This class provides methods for invoking the Anthropic API to generate responses
based on given input data. It extends the Client class.

### \_\_init\_\_

```
__init__(
    api_key,
    model="claude-3-5-sonnet-latest",
    system_prompt="",
    temperature=None,
    cache=None,
)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str` | The API key for the Anthropic API. | *required* |
| `model` | `str` | The model to use for the Anthropic API. | `'claude-3-5-sonnet-latest'` |
| `system_prompt` | `str` | The system prompt to use for the Anthropic API. | `''` |
| `temperature` | `float | None` | The temperature to use for the Anthropic API. | `None` |
| `cache` | `Cache | None` | The cache to use for the Anthropic API. | `None` |

## Usage example

```
from datapizza.clients.anthropic import AnthropicClient

client = AnthropicClient(
    api_key="YOUR_API_KEY"
    model="claude-3-5-sonnet-20240620",
)
resposne = client.invoke("hi")
print(response.text)
```

## Show thinking

```
import os

from datapizza.clients.anthropic import AnthropicClient
from dotenv import load_dotenv

load_dotenv()

client = AnthropicClient(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-0",
)

response = client.invoke("Hi", thinking =  {"type": "enabled", "budget_tokens": 1024})
print(response)
```

---

<a id="22"></a>

## Mistral - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/Avaiable_Clients/mistral/

# Mistral

```
pip install datapizza-ai-clients-mistral
```

## datapizza.clients.mistral.MistralClient

Bases: `Client`

A client for interacting with the Mistral API.

This class provides methods for invoking the Mistral API to generate responses
based on given input data. It extends the Client class.

### \_\_init\_\_

```
__init__(
    api_key,
    model="mistral-large-latest",
    system_prompt="",
    temperature=None,
    cache=None,
)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str` | The API key for the Mistral API. | *required* |
| `model` | `str` | The model to use for the Mistral API. | `'mistral-large-latest'` |
| `system_prompt` | `str` | The system prompt to use for the Mistral API. | `''` |
| `temperature` | `float | None` | The temperature to use for the Mistral API. | `None` |
| `cache` | `Cache | None` | The cache to use for the Mistral API. | `None` |

# Usage Example

```
from datapizza.clients.mistral import MistralClient

client = MistralClient(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model="mistral-small-latest",
    system_prompt="You are a helpful assistant that responds short and concise.",
)
response = client.invoke("hi")
print(response.text)
```

---

<a id="23"></a>

## Openai like - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Clients/Avaiable_Clients/openai-like/

# Openai like

```
pip install datapizza-ai-clients-openai-like
```

## datapizza.clients.openai\_like.OpenAILikeClient

Bases: `Client`

A client for interacting with the OpenAI API.

This class provides methods for invoking the OpenAI API to generate responses
based on given input data. It extends the Client class.

## Key Differences from OpenAIClient

The main difference between OpenAILikeClient and OpenAIClient is the API endpoint they use:

* OpenAILikeClient uses the chat completions API
* OpenAIClient uses the responses API

This makes OpenAILikeClient compatible with services that implement the OpenAI-compatible completions API, such as local models served through Ollama or other providers that follow the OpenAI API specification but only support the completions endpoint.

## Usage example

```
from datapizza.clients.openai_like import OpenAILikeClient

client = OpenAILikeClient(
    api_key="OPENAI_API_KEY",
    system_prompt="You are a helpful assistant.",
)

response = client.invoke("What is the capital of France?")
print(response.content)
```

---


## Agents

<a id="16"></a>

## Agent - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Agents/agent/

# Agent

## datapizza.agents.agent.Agent

### \_\_init\_\_

```
__init__(
    name=None,
    client=None,
    *,
    system_prompt=None,
    tools=None,
    max_steps=None,
    terminate_on_text=True,
    stateless=True,
    gen_args=None,
    memory=None,
    stream=None,
    can_call=None,
    logger=None,
    planning_interval=0,
    planning_prompt=PLANNING_PROMT,
)
```

Initialize the agent.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `name` | `str` | The name of the agent. Defaults to None. | `None` |
| `client` | `Client` | The client to use for the agent. Defaults to None. | `None` |
| `system_prompt` | `str` | The system prompt to use for the agent. Defaults to None. | `None` |
| `tools` | `list[Tool]` | A list of tools to use with the agent. Defaults to None. | `None` |
| `max_steps` | `int` | The maximum number of steps to execute. Defaults to None. | `None` |
| `terminate_on_text` | `bool` | Whether to terminate the agent on text. Defaults to True. | `True` |
| `stateless` | `bool` | Whether to use stateless execution. Defaults to True. | `True` |
| `gen_args` | `dict[str, Any]` | Additional arguments to pass to the agent's execution. Defaults to None. | `None` |
| `memory` | `Memory` | The memory to use for the agent. Defaults to None. | `None` |
| `stream` | `bool` | Whether to stream the agent's execution. Defaults to None. | `None` |
| `can_call` | `list[Agent]` | A list of agents that can call the agent. Defaults to None. | `None` |
| `logger` | `AgentLogger` | The logger to use for the agent. Defaults to None. | `None` |
| `planning_interval` | `int` | The planning interval to use for the agent. Defaults to 0. | `0` |
| `planning_prompt` | `str` | The planning prompt to use for the agent planning steps. Defaults to PLANNING\_PROMT. | `PLANNING_PROMT` |

### a\_run `async`

```
a_run(task_input, tool_choice='auto', **gen_kwargs)
```

Run the agent on a task input asynchronously.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `task_input` | `str` | The input text/prompt to send to the model | *required* |
| `tool_choice` | `Literal['auto', 'required', 'none', 'required_first'] | list[str]` | Controls which tool to use ("auto" by default) | `'auto'` |
| `**gen_kwargs` |  | Additional keyword arguments to pass to the agent's execution | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `StepResult | None` | The final result of the agent's execution |

### a\_stream\_invoke `async`

```
a_stream_invoke(
    task_input, tool_choice="auto", **gen_kwargs
)
```

Stream the agent's execution asynchronously, yielding intermediate steps and final result.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `task_input` | `str` | The input text/prompt to send to the model | *required* |
| `tool_choice` | `Literal['auto', 'required', 'none', 'required_first'] | list[str]` | Controls which tool to use ("auto" by default) | `'auto'` |
| `**gen_kwargs` |  | Additional keyword arguments to pass to the agent's execution | `{}` |

Yields:

| Type | Description |
| --- | --- |
| `AsyncGenerator[ClientResponse | StepResult | Plan | None]` | The intermediate steps and final result of the agent's execution |

### run

```
run(task_input, tool_choice='auto', **gen_kwargs)
```

Run the agent on a task input.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `task_input` | `str` | The input text/prompt to send to the model | *required* |
| `tool_choice` | `Literal['auto', 'required', 'none', 'required_first'] | list[str]` | Controls which tool to use ("auto" by default) | `'auto'` |
| `**gen_kwargs` |  | Additional keyword arguments to pass to the agent's execution | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `StepResult | None` | The final result of the agent's execution |

### stream\_invoke

```
stream_invoke(task_input, tool_choice='auto', **gen_kwargs)
```

Stream the agent's execution, yielding intermediate steps and final result.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `task_input` | `str` | The input text/prompt to send to the model | *required* |
| `tool_choice` | `Literal['auto', 'required', 'none', 'required_first'] | list[str]` | Controls which tool to use ("auto" by default) | `'auto'` |
| `**gen_kwargs` |  | Additional keyword arguments to pass to the agent's execution | `{}` |

Yields:

| Type | Description |
| --- | --- |
| `ClientResponse | StepResult | Plan | None` | The intermediate steps and final result of the agent's execution |

---


## Embedders

<a id="26"></a>

## ChunkEmbedder - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Embedders/chunk_embedder/

# ChunkEmbedder

## datapizza.embedders.ChunkEmbedder

Bases: `PipelineComponent`

ChunkEmbedder is a module that given a list of chunks, it put a list of embeddings in each chunk.

### \_\_init\_\_

```
__init__(
    client,
    model_name=None,
    embedding_name=None,
    batch_size=2047,
)
```

Initialize the ChunkEmbedder.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `client` | `BaseEmbedder` | The client to use for embedding. | *required* |
| `model_name` | `str` | The model name to use for embedding. Defaults to None. | `None` |
| `embedding_name` | `str` | The name of the embedding to use. Defaults to None. | `None` |
| `batch_size` | `int` | The batch size to use for embedding. Defaults to 2047. | `2047` |

### a\_embed `async`

```
a_embed(nodes)
```

Asynchronously embeds the given list of chunks.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nodes` | `list[Chunk]` | The list of chunks to embed. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | list[Chunk]: The list of chunks with embeddings. |

### embed

```
embed(nodes)
```

Embeds the given list of chunks.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nodes` | `list[Chunk]` | The list of chunks to embed. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | list[Chunk]: The list of chunks with embeddings. |

## Usage

```
from datapizza.embedders import ChunkEmbedder
from datapizza.core.clients import Client

# Initialize with any compatible client
client = Client(...)  # Your client instance
embedder = ChunkEmbedder(
    client=client,
    model_name="text-embedding-ada-002",  # Optional model override
    embedding_name="my_embeddings",       # Optional custom embedding name
    batch_size=100                        # Optional batch size for processing
)

# Embed chunks - adds embeddings to chunk objects
embedded_chunks = embedder.embed(chunks)
```

## Features

* Specialized for embedding lists of Chunk objects
* Batch processing with configurable batch size
* Adds embeddings directly to Chunk objects
* Preserves original chunk structure and metadata
* Async embedding support with `a_embed()`
* Memory efficient batch processing
* Works with any compatible LLM client

## Examples

### Basic Chunk Embedding

```
import os

from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.type import Chunk
from dotenv import load_dotenv

load_dotenv()

# Create client and embedder
client = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))
embedder = ChunkEmbedder(
    client=client,
    model_name="text-embedding-ada-002",
    batch_size=50
)

# Create sample chunks
chunks = [
    Chunk(id="1", text="First chunk of text", metadata={"source": "doc1"}),
    Chunk(id="2", text="Second chunk of text", metadata={"source": "doc2"}),
    Chunk(id="3", text="Third chunk of text", metadata={"source": "doc3"})
]

# Embed chunks (modifies chunks in-place)
embedded_chunks = embedder.embed(chunks)

# Check embeddings were added
for i, chunk in enumerate(embedded_chunks):
    print(f"Chunk {i+1}:")
    print(f"  Text: {chunk.text[:50]}...")
    print(f"  Embeddings: {len(chunk.embeddings)}")
    if chunk.embeddings:
        print(f"  Embedding name: {chunk.embeddings[0].name}")
        print(f"  Vector size: {len(chunk.embeddings[0].vector)}")
```

---

<a id="27"></a>

## CohereEmbedder - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Embedders/cohere_embedder/

# CohereEmbedder

```
pip install datapizza-ai-embedders-cohere
```

## datapizza.embedders.cohere.CohereEmbedder

Bases: `BaseEmbedder`

## Usage

```
from datapizza.embedders.cohere import CohereEmbedder

embedder = CohereEmbedder(
    api_key="your-cohere-api-key",
    base_url="https://api.cohere.ai/v1",
    input_type="search_document"  # or "search_query"
)

# Embed a single text
embedding = embedder.embed("Hello world", model_name="embed-english-v3.0")

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"],
    model_name="embed-english-v3.0"
)
```

## Features

* Supports Cohere's embedding models
* Configurable input type for search documents or queries
* Handles both single text and batch text embedding
* Async embedding support with `a_embed()`
* Custom endpoint support for compatible APIs
* Uses Cohere's ClientV2 for optimal performance

## Examples

### Basic Text Embedding

```
from datapizza.embedders.cohere import CohereEmbedder

embedder = CohereEmbedder(
    api_key="your-cohere-api-key",
    base_url="https://api.cohere.ai/v1",
    input_type="search_document"
)

# Single text embedding
text = "This is a sample document for embedding."
embedding = embedder.embed(text, model_name="embed-english-v3.0")

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Search Query Embedding

```
from datapizza.embedders.cohere import CohereEmbedder

# Configure for search queries
embedder = CohereEmbedder(
    api_key="your-cohere-api-key",
    base_url="https://api.cohere.ai/v1",
    input_type="search_query"
)

query = "What is machine learning?"
embedding = embedder.embed(query, model_name="embed-english-v3.0")

print(f"Query embedding size: {len(embedding)}")
```

### Batch Text Embedding

```
from datapizza.embedders.cohere import CohereEmbedder

embedder = CohereEmbedder(
    api_key="your-cohere-api-key",
    base_url="https://api.cohere.ai/v1"
)

texts = [
    "First document to embed",
    "Second document to embed",
    "Third document to embed"
]

embeddings = embedder.embed(texts, model_name="embed-english-v3.0")

for i, emb in enumerate(embeddings):
    print(f"Document {i+1} embedding size: {len(emb)}")
```

### Async Embedding

```
import asyncio
from datapizza.embedders.cohere import CohereEmbedder

async def embed_async():
    embedder = CohereEmbedder(
        api_key="your-cohere-api-key",
        base_url="https://api.cohere.ai/v1"
    )

    text = "Async embedding example"
    embedding = await embedder.a_embed(text, model_name="embed-english-v3.0")

    return embedding

# Run async function
embedding = asyncio.run(embed_async())
```

---

<a id="28"></a>

## FastEmbedder - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Embedders/fast_embedder/

# FastEmbedder

```
pip install datapizza-ai-embedders-fastembedder
```

## datapizza.embedders.fastembedder.FastEmbedder

Bases: `BaseEmbedder`

## Usage

```
from datapizza.embedders.fastembedder import FastEmbedder

embedder = FastEmbedder(
    model_name="Qdrant/bm25",
    embedding_name="bm25_embeddings",
)

# Embed text (returns sparse embeddings)
embeddings = embedder.embed(["Hello world", "Another text"])
print(embeddings)
```

## Features

* Uses FastEmbed for efficient sparse text embeddings
* Local model execution (no API calls required)
* Configurable model caching directory
* Custom embedding naming
* Sparse embedding format for memory efficiency
* Both sync and async embedding support

---

<a id="29"></a>

## GoogleEmbedder - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Embedders/google_embedder/

# GoogleEmbedder

```
pip install datapizza-ai-embedders-google
```

## datapizza.embedders.google.GoogleEmbedder

Bases: `BaseEmbedder`

## Usage

```
from datapizza.embedders.google import GoogleEmbedder

embedder = GoogleEmbedder(
    api_key="your-google-api-key"
)

# Embed a single text
embedding = embedder.embed("Hello world", model_name="models/embedding-001")

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"],
    model_name="models/embedding-001"
)
```

## Features

* Supports Google's Gemini embedding models
* Handles both single text and batch text embedding
* Async embedding support with `a_embed()`
* Automatic client initialization and management
* Uses Google's Generative AI SDK

## Examples

### Basic Text Embedding

```
from datapizza.embedders.google import GoogleEmbedder

embedder = GoogleEmbedder(api_key="your-google-api-key")

# Single text embedding
text = "This is a sample document for embedding."
embedding = embedder.embed(text, model_name="models/embedding-001")

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Async Embedding

```
import asyncio
from datapizza.embedders.google import GoogleEmbedder

async def embed_async():
    embedder = GoogleEmbedder(api_key="your-google-api-key")

    text = "Async embedding example"
    embedding = await embedder.a_embed(text, model_name="models/embedding-001")

    return embedding

# Run async function
embedding = asyncio.run(embed_async())
```

---

<a id="30"></a>

## OllamaEmbedder - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Embedders/ollama_embedder/

# OllamaEmbedder

Ollama embedders are OpenAI-compatible, which means you can use the OpenAI embedder to generate embeddings with Ollama models. Simply configure the OpenAI embedder with Ollama's base URL and leave the API key empty.

## Usage

```
from datapizza.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key="",
    base_url="http://localhost:11434/v1",
)

# Embed a single text
embedding = embedder.embed("Hello world", model_name="nomic-embed-text")

print(embedding)

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"], model_name="nomic-embed-text"
)

print(embeddings)
```

---

<a id="31"></a>

## OpenAIEmbedder - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Embedders/openai_embedder/

# OpenAIEmbedder

## datapizza.embedders.openai.OpenAIEmbedder

Bases: `BaseEmbedder`

## Usage

```
from datapizza.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key="your-openai-api-key",
    base_url="https://api.openai.com/v1"  # Optional custom base URL
)

# Embed a single text
embedding = embedder.embed("Hello world", model_name="text-embedding-ada-002")

# Embed multiple texts
embeddings = embedder.embed(
    ["Hello world", "Another text"],
    model_name="text-embedding-ada-002"
)
```

## Features

* Supports OpenAI's embedding models
* Handles both single text and batch text embedding
* Async embedding support with `a_embed()`
* Custom base URL support for compatible APIs
* Automatic client initialization and management

## Examples

### Basic Text Embedding

```
from datapizza.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(api_key="your-api-key")

# Single text embedding
text = "This is a sample document for embedding."
embedding = embedder.embed(text, model_name="text-embedding-ada-002")

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Async Embedding

```
import asyncio
from datapizza.embedders.openai import OpenAIEmbedder

async def embed_async():
    embedder = OpenAIEmbedder(api_key="your-api-key")

    text = "Async embedding example"
    embedding = await embedder.a_embed(text, model_name="text-embedding-ada-002")

    return embedding

# Run async function
embedding = asyncio.run(embed_async())
```

---


## Vectorstore

<a id="65"></a>

## Milvus - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Vectorstore/milvus_vectorstore/

# Milvus

```
pip install datapizza-ai-vectorstores-milvus
```

## Usage

```
from datapizza.vectorstores.milvus import MilvusVectorstore

# Option A) Milvus Server / Zilliz Cloud
vectorstore = MilvusVectorstore(
    host="localhost",
    port=19530,
    # user="username",            # Optional
    # password="password",        # Optional
    # secure=True,                # Optional (TLS)
    # token="zilliz_token",       # Optional (Zilliz)
)

# Option B) Single-URI style (works with Milvus, Zilliz, or Milvus Lite)
vectorstore = MilvusVectorstore(uri="./milvus.db")  # Milvus Lite
```

## Features

* Connect via `host/port` or a single `uri` (supports Milvus Server, Zilliz Cloud, and Milvus Lite).
* Works with **dense** and **sparse** embeddings in the *same* collection.
* Named vector fields for **multi-vector** collections.
* Batch/async operations: `add` / `a_add`, `search` / `a_search`.
* Collection management: `create_collection`, `delete_collection`, `get_collections`.
* Entities ops: `retrieve`, `update` (upsert), `remove`.
* Flexible indexing (defaults provided; accepts custom `IndexParams`).
* Dynamic metadata via Milvus’ `$meta` (stored from `Chunk.metadata`).

---

<a id="64"></a>

## Qdrant - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Vectorstore/qdrant_vectorstore/

# Qdrant

```
pip install datapizza-ai-vectorstores-qdrant
```

## datapizza.vectorstores.qdrant.QdrantVectorstore

Bases: `Vectorstore`

datapizza-ai implementation of a Qdrant vectorstore.

### \_\_init\_\_

```
__init__(host=None, port=6333, api_key=None, **kwargs)
```

Initialize the QdrantVectorstore.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `host` | `str` | The host to use for the Qdrant client. Defaults to None. | `None` |
| `port` | `int` | The port to use for the Qdrant client. Defaults to 6333. | `6333` |
| `api_key` | `str` | The API key to use for the Qdrant client. Defaults to None. | `None` |
| `**kwargs` |  | Additional keyword arguments to pass to the Qdrant client. | `{}` |

### a\_search `async`

```
a_search(
    collection_name,
    query_vector,
    k=10,
    vector_name=None,
    **kwargs,
)
```

Search for chunks in a collection by their query vector.

### add

```
add(chunk, collection_name=None)
```

Add a single chunk or list of chunks to the vectorstore.
Args:
chunk (Chunk | list[Chunk]): The chunk or list of chunks to add.
collection\_name (str, optional): The name of the collection to add the chunks to. Defaults to None.

### create\_collection

```
create_collection(collection_name, vector_config, **kwargs)
```

Create a new collection in Qdrant if it doesn't exist with the specified vector configurations

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `collection_name` | `str` | Name of the collection to create | *required* |
| `vector_config` | `list[VectorConfig]` | List of vector configurations specifying dimensions and distance metrics | *required* |
| `**kwargs` |  | Additional arguments to pass to Qdrant's create\_collection | `{}` |

### delete\_collection

```
delete_collection(collection_name, **kwargs)
```

Delete a collection in Qdrant.

### dump\_collection

```
dump_collection(
    collection_name, page_size=100, with_vectors=False
)
```

Dumps all points from a collection in a chunk-wise manner.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `collection_name` | `str` | Name of the collection to dump. | *required* |
| `page_size` | `int` | Number of points to retrieve per batch. | `100` |
| `with_vectors` | `bool` | Whether to include vectors in the dumped chunks. | `False` |

Yields:

| Name | Type | Description |
| --- | --- | --- |
| `Chunk` | `Chunk` | A chunk object from the collection. |

### get\_collections

```
get_collections()
```

Get all collections in Qdrant.

### remove

```
remove(collection_name, ids, **kwargs)
```

Remove chunks from a collection by their IDs.
Args:
collection\_name (str): The name of the collection to remove the chunks from.
ids (list[str]): The IDs of the chunks to remove.
\*\*kwargs: Additional keyword arguments to pass to the Qdrant client.

### retrieve

```
retrieve(collection_name, ids, **kwargs)
```

Retrieve chunks from a collection by their IDs.
Args:
collection\_name (str): The name of the collection to retrieve the chunks from.
ids (list[str]): The IDs of the chunks to retrieve.
\*\*kwargs: Additional keyword arguments to pass to the Qdrant client.
Returns:
list[Chunk]: The list of chunks retrieved from the collection.

### search

```
search(
    collection_name,
    query_vector,
    k=10,
    vector_name=None,
    **kwargs,
)
```

Search for chunks in a collection by their query vector.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `collection_name` | `str` | The name of the collection to search in. | *required* |
| `query_vector` | `list[float]` | The query vector to search for. | *required* |
| `k` | `int` | The number of results to return. Defaults to 10. | `10` |
| `vector_name` | `str` | The name of the vector to search for. Defaults to None. | `None` |
| `**kwargs` |  | Additional keyword arguments to pass to the Qdrant client. | `{}` |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | list[Chunk]: The list of chunks found in the collection. |

## Usage

```
from datapizza.vectorstores.qdrant import QdrantVectorstore

# Connect to Qdrant server
vectorstore = QdrantVectorstore(
    host="localhost",
    port=6333,
    api_key="your-api-key"  # Optional
)

# Or use in-memory/file storage
vectorstore = QdrantVectorstore(
    location=":memory:"  # Or path to file
)
```

## Features

* Connect to Qdrant server or use local storage
* Support for both dense and sparse embeddings
* Named vector configurations for multi-vector collections
* Batch operations for efficient processing
* Collection management (create, delete, list)
* Chunk-based operations with metadata preservation
* Async support for all operations
* Point-level operations (add, update, remove, retrieve)

## Examples

### Basic Setup and Collection Creation

```
from datapizza.core.vectorstore import Distance, VectorConfig
from datapizza.type import EmbeddingFormat
from datapizza.vectorstores.qdrant import QdrantVectorstore

vectorstore = QdrantVectorstore(location=":memory:")

# Create collection with vector configuration
vector_config = [
    VectorConfig(
        name="text_embeddings",
        dimensions=3,
        format=EmbeddingFormat.DENSE,
        distance=Distance.COSINE
    )
]

vectorstore.create_collection(
    collection_name="documents",
    vector_config=vector_config
)

# Add nodes and search

import uuid
from datapizza.type import Chunk, DenseEmbedding
from datapizza.vectorstores.qdrant import QdrantVectorstore

# Create chunks with embeddings
chunks = [
    Chunk(
        id=str(uuid.uuid4()),
        text="First document content",
        metadata={"source": "doc1.txt"},
        embeddings=[DenseEmbedding(name="text_embeddings", vector=[0.1, 0.2, 0.3])]
    ),
    Chunk(
        id=str(uuid.uuid4()),
        text="Second document content",
        metadata={"source": "doc2.txt"},
        embeddings=[DenseEmbedding(name="text_embeddings", vector=[0.4, 0.5, 0.6])]
    )
]

# Add chunks to collection
vectorstore.add(chunks, collection_name="documents")

# Search for similar chunks
query_vector = [0.1, 0.2, 0.3]
results = vectorstore.search(
    collection_name="documents",
    query_vector=query_vector,
    k=5
)

for chunk in results:
    print(f"Text: {chunk.text}")
    print(f"Metadata: {chunk.metadata}")
```

---


## Memory

<a id="32"></a>

## Memory - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/memory/

# Memory

## datapizza.memory.memory.Memory

A class for storing the memory of a chat, organized by conversation turns.
Each turn can contain multiple blocks (text, function calls, or structured data).

### \_\_bool\_\_

```
__bool__()
```

Return True if memory contains any turns, False otherwise.

### \_\_delitem\_\_

```
__delitem__(index)
```

Delete a specific turn.

### \_\_eq\_\_

```
__eq__(other)
```

Compare two Memory objects based on their content hash.
This is more efficient than comparing the full content structure.

### \_\_getitem\_\_

```
__getitem__(index)
```

Get all blocks from a specific turn.

### \_\_hash\_\_

```
__hash__()
```

Creates a deterministic hash based on the content of memory turns.

### \_\_iter\_\_

```
__iter__()
```

Iterate through all blocks in all turns.

### \_\_len\_\_

```
__len__()
```

Return the total number of turns.

### \_\_repr\_\_

```
__repr__()
```

Return a detailed string representation of the memory.

### \_\_setitem\_\_

```
__setitem__(index, value)
```

Set blocks for a specific turn.

### \_\_str\_\_

```
__str__()
```

Return a string representation of the memory.

### add\_to\_last\_turn

```
add_to_last_turn(block)
```

Add a block to the most recent turn. Creates a new turn if memory is empty.
Args:
block (Block): The block to add to the most recent turn.

### add\_turn

```
add_turn(blocks, role)
```

Add a new conversation turn containing one or more blocks.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `blocks` | `list[Block] | Block` | The blocks to add to the new turn. | *required* |
| `role` | `ROLE` | The role of the new turn. | *required* |

### clear

```
clear()
```

Clear all memory.

### copy

```
copy()
```

Deep copy the memory.

### iter\_blocks

```
iter_blocks()
```

Iterate through blocks.

### json\_dumps

```
json_dumps()
```

Serialize the memory to JSON.

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `str` | `str` | The JSON representation of the memory. |

### json\_loads

```
json_loads(json_str)
```

Deserialize JSON to the memory.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `json_str` | `str` | The JSON string to deserialize. | *required* |

### new\_turn

```
new_turn(role=ROLE.ASSISTANT)
```

Add a new conversation turn.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `role` | `ROLE` | The role of the new turn. Defaults to ROLE.ASSISTANT. | `ASSISTANT` |

### to\_dict

```
to_dict()
```

Convert memory to a dictionary.

Returns:

| Type | Description |
| --- | --- |
| `list[dict]` | list[dict]: The dictionary representation of the memory. |

---


## Type

<a id="59"></a>

## Blocks - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Type/block/

# Blocks

## datapizza.type.Block

A class for storing the response from a client.

### to\_dict `abstractmethod`

```
to_dict()
```

Convert the block to a dictionary for JSON serialization.

## datapizza.type.TextBlock

Bases: `Block`

A class for storing the text response from a client.

### \_\_init\_\_

```
__init__(content, type='text')
```

Initialize a TextBlock object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `content` | `str` | The content of the text block. | *required* |
| `type` | `str` | The type of the text block. Defaults to "text". | `'text'` |

## datapizza.type.MediaBlock

Bases: `Block`

A class for storing the media response from a client.

### \_\_init\_\_

```
__init__(media, type='media')
```

Initialize a MediaBlock object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `media` | `Media` | The media of the media block. | *required* |
| `type` | `str` | The type of the media block. Defaults to "media". | `'media'` |

## datapizza.type.ThoughtBlock

Bases: `Block`

A class for storing the thought from a client.

### \_\_init\_\_

```
__init__(content, type='thought')
```

Initialize a ThoughtBlock object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `content` | `str` | The content of the thought block. | *required* |
| `type` | `str` | The type of the thought block. Defaults to "thought". | `'thought'` |

## datapizza.type.FunctionCallBlock

Bases: `Block`

A class for storing the function call from a client.

### \_\_init\_\_

```
__init__(id, arguments, name, tool, type='function')
```

Initialize a FunctionCallBlock object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `id` | `str` | The id of the function call block. | *required* |
| `arguments` | `dict[str, Any]` | The arguments of the function call block. | *required* |
| `name` | `str` | The name of the function call block. | *required* |
| `tool` | `Tool` | The tool of the function call block. | *required* |

## datapizza.type.FunctionCallResultBlock

Bases: `Block`

A class for storing the function call response from a client.

### \_\_init\_\_

```
__init__(id, tool, result, type='function_call_result')
```

Initialize a FunctionCallResultBlock object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `id` | `str` | The id of the function call result block. | *required* |
| `tool` | `Tool` | The tool of the function call result block. | *required* |
| `result` | `str` | The result of the function call result block. | *required* |

## datapizza.type.StructuredBlock

Bases: `Block`

A class for storing the structured response from a client.

### \_\_init\_\_

```
__init__(content, type='structured')
```

Initialize a StructuredBlock object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `content` | `BaseModel` | The content of the structured block. | *required* |
| `type` | `str` | The type of the structured block. Defaults to "structured". | `'structured'` |

---

<a id="60"></a>

## Chunk - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Type/chunk/

# Chunk

## datapizza.type.Chunk `dataclass`

A class for storing the chunk response from a client.

### \_\_init\_\_

```
__init__(id, text, embeddings=None, metadata=None)
```

Initialize a Chunk object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `id` | `str` | The id of the chunk. | *required* |
| `text` | `str` | The text of the chunk. | *required* |
| `embeddings` | `list[Embedding]` | The embeddings of the chunk. Defaults to []. | `None` |
| `metadata` | `dict` | The metadata of the chunk. Defaults to {}. | `None` |

## Overview

The `Chunk` class represents a unit of text content that has been segmented from a larger document. It's a fundamental data structure in datapizza-ai used throughout the RAG pipeline for text processing, embedding, and retrieval operations.
**Serializable**: Can be easily stored and retrieved from databases

---

<a id="61"></a>

## Media - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Type/media/

# Media

## datapizza.type.Media

A class for storing the media response from a client.

### \_\_init\_\_

```
__init__(
    *,
    extension=None,
    media_type,
    source_type,
    source,
    detail="high",
)
```

A class for storing the media response from a client.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `extension` | `str` | The file extension of the media. Defaults to None. | `None` |
| `media_type` | `Literal['image', 'video', 'audio', 'pdf']` | The type of media. Defaults to "image". | *required* |
| `source_type` | `Literal['url', 'base64', 'path', 'pil', 'raw']` | The source type of the media. Defaults to "url". | *required* |
| `source` | `Any` | The source of the media. Defaults to None. | *required* |

### to\_dict

```
to_dict()
```

Convert the media to a dictionary for JSON serialization.

---

<a id="62"></a>

## Node - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Type/node/

# Node

## datapizza.type.Node

Class representing a node in a document graph.

### content `property`

```
content
```

Get the textual content of this node and its children.

### is\_leaf `property`

```
is_leaf
```

Check if the node is a leaf node (has no children).

### \_\_eq\_\_

```
__eq__(other)
```

Check if two nodes are equal.

### \_\_hash\_\_

```
__hash__()
```

Hash the node.

### \_\_init\_\_

```
__init__(
    children=None,
    metadata=None,
    node_type=NodeType.SECTION,
    content=None,
)
```

Initialize a Node object.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `children` | `list[Node] | None` | List of child nodes | `None` |
| `metadata` | `dict | None` | Dictionary of metadata | `None` |
| `content` | `str | None` | Content object for leaf nodes | `None` |

### add\_child

```
add_child(child)
```

Add a child node to this node.

### remove\_child

```
remove_child(child)
```

Remove a child node from this node.

## datapizza.type.MediaNode

Bases: `Node`

Class representing a media node in a document graph.

---

<a id="63"></a>

## Tool - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Type/tool/

# Tool

## datapizza.tools.Tool

Class that wraps a function while preserving its behavior and adding attributes.

### \_\_init\_\_

```
__init__(
    func=None,
    name=None,
    description=None,
    end=False,
    properties=None,
    required=None,
    strict=False,
)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `func` | `Callable | None` | The function to wrap. | `None` |
| `name` | `str | None` | The name of the tool. | `None` |
| `description` | `str | None` | The description of the tool. | `None` |
| `end` | `bool` | Whether the tool ends a chain of operations. | `False` |
| `properties` | `dict[str, dict[str, Any]] | None` | The properties of the tool. | `None` |
| `required` | `list[str] | None` | The required parameters of the tool. | `None` |
| `strict` | `bool` | Whether the tool is strict. | `False` |

### to\_dict

```
to_dict()
```

Convert the tool to a dictionary for JSON serialization.

---


## Pipelines

<a id="52"></a>

## Dag - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Pipelines/dag/

# Dag

## datapizza.pipeline.dag\_pipeline.DagPipeline

A pipeline that runs a graph of a dependency graph.

### a\_run `async`

```
a_run(data)
```

Run the pipeline asynchronously.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data` | `dict` | The input data to the pipeline. | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `dict` |  | The results of the pipeline. |

### add\_module

```
add_module(node_name, node)
```

Add a module to the pipeline.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `node_name` | `str` | The name of the module. | *required* |
| `node` | `PipelineComponent` | The module to add. | *required* |

### connect

```
connect(
    source_node, target_node, target_key, source_key=None
)
```

Connect two nodes in the pipeline.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source_node` | `str` | The name of the source node. | *required* |
| `target_node` | `str` | The name of the target node. | *required* |
| `target_key` | `str` | The key to store the result of the target node in the source node. | *required* |
| `source_key` | `str` | The key to retrieve the result of the source node from the target node. Defaults to None. | `None` |

### from\_yaml

```
from_yaml(config_path)
```

Load the pipeline from a YAML configuration file.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `config_path` | `str` | Path to the YAML configuration file. | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `DagPipeline` | `DagPipeline` | The pipeline instance. |

### run

```
run(data)
```

Run the pipeline.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data` | `dict` | The input data to the pipeline. | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `dict` | `dict` | The results of the pipeline. |

---

<a id="53"></a>

## Functional - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Pipelines/functional/

# Functional

## datapizza.pipeline.functional\_pipeline.FunctionalPipeline

Pipeline for executing a series of nodes with dependencies.

### branch

```
branch(condition, if_true, if_false, dependencies=None)
```

Branch execution based on a condition.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `condition` | `Callable` | The condition to evaluate. | *required* |
| `if_true` | `FunctionalPipeline` | The pipeline to execute if the condition is True. | *required* |
| `if_false` | `FunctionalPipeline` | The pipeline to execute if the condition is False. | *required* |
| `dependencies` | `list[Dependency]` | List of dependencies for the node. Defaults to None. | `None` |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `FunctionalPipeline` | `FunctionalPipeline` | The pipeline instance. |

### execute

```
execute(initial_data=None, context=None)
```

Execute the pipeline and return the results.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `initial_data` | `dict[str, Any] | None` | Dictionary where keys are node names and values are the data to be passed to those nodes when they execute. | `None` |
| `context` | `dict | None` | Dictionary where keys are node names and values are the data to be passed to those nodes when they execute. | `None` |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `dict` | `dict[str, Any]` | The results of the pipeline. |

### foreach

```
foreach(name, do, dependencies=None)
```

Execute a sub-pipeline for each item in a collection.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `name` | `str` | The name of the node. | *required* |
| `do` | `PipelineComponent` | The sub-pipeline to execute for each item. | *required* |
| `dependencies` | `list[Dependency]` | List of dependencies for the node. Defaults to None. | `None` |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `FunctionalPipeline` | `FunctionalPipeline` | The pipeline instance. |

### from\_yaml `staticmethod`

```
from_yaml(yaml_path)
```

Constructs a FunctionalPipeline from a YAML configuration file.
The YAML should contain 'modules' (optional) defining reusable components
and 'pipeline' defining the sequence of steps.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `yaml_path` | `str` | Path to the YAML configuration file. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `FunctionalPipeline` | A configured FunctionalPipeline instance. |

Raises:

| Type | Description |
| --- | --- |
| `ValueError` | If the YAML format is invalid, a module cannot be loaded, or a referenced node/condition name is not found. |
| `KeyError` | If a required key is missing in the YAML structure. |
| `FileNotFoundError` | If the yaml\_path does not exist. |
| `YAMLError` | If the YAML file cannot be parsed. |
| `ImportError` | If a specified module cannot be imported. |
| `AttributeError` | If a specified class/function is not found in the module. |

### get

```
get(name)
```

Get the result of a node.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `name` | `str` | The name of the node. | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `FunctionalPipeline` | `FunctionalPipeline` | The pipeline instance. |

### run

```
run(name, node, dependencies=None, kwargs=None)
```

Add a node to the pipeline with optional dependencies.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `name` | `str` | The name of the node. | *required* |
| `node` | `PipelineComponent` | The node to add. | *required* |
| `dependencies` | `list[Dependency]` | List of dependencies for the node. Defaults to None. | `None` |
| `kwargs` | `dict[str, Any]` | Additional keyword arguments to pass to the node. Defaults to None. | `None` |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `FunctionalPipeline` | `FunctionalPipeline` | The pipeline instance. |

### then

```
then(
    name, node, target_key, dependencies=None, kwargs=None
)
```

Add a node to execute after the previous node.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `name` | `str` | The name of the node. | *required* |
| `node` | `PipelineComponent` | The node to add. | *required* |
| `target_key` | `str` | The key to store the result of the node in the previous node. | *required* |
| `dependencies` | `list[Dependency]` | List of dependencies for the node. Defaults to None. | `None` |
| `kwargs` | `dict[str, Any]` | Additional keyword arguments to pass to the node. Defaults to None. | `None` |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `FunctionalPipeline` | `FunctionalPipeline` | The pipeline instance. |

---

<a id="51"></a>

## Ingestion - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Pipelines/ingestion/

# Ingestion

## datapizza.pipeline.pipeline.IngestionPipeline

A pipeline for ingesting data into a vector store.

### \_\_init\_\_

```
__init__(
    modules=None, vector_store=None, collection_name=None
)
```

Initialize the ingestion pipeline.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `modules` | `list[PipelineComponent]` | List of pipeline components. Defaults to None. | `None` |
| `vector_store` | `Vectorstore` | Vector store to store the ingested data. Defaults to None. | `None` |
| `collection_name` | `str` | Name of the vector store collection to store the ingested data. Defaults to None. | `None` |

### a\_run `async`

```
a_run(file_path, metadata=None)
```

Run the ingestion pipeline asynchronously.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file_path` | `str | list[str]` | The file path or list of file paths to ingest. | *required* |
| `metadata` | `dict` | Metadata to add to the ingested chunks. Defaults to None. | `None` |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk] | None` | list[Chunk] | None: If vector\_store is not set, returns all accumulated chunks from all files. If vector\_store is set, returns None after storing all chunks. |

### from\_yaml

```
from_yaml(config_path)
```

Load the ingestion pipeline from a YAML configuration file.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `config_path` | `str` | Path to the YAML configuration file. | *required* |

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `IngestionPipeline` | `IngestionPipeline` | The ingestion pipeline instance. |

### run

```
run(file_path, metadata=None)
```

Run the ingestion pipeline.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file_path` | `str | list[str]` | The file path or list of file paths to ingest. | *required* |
| `metadata` | `dict` | Metadata to add to the ingested chunks. Defaults to None. | `None` |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk] | None` | list[Chunk] | None: If vector\_store is not set, returns all accumulated chunks from all files. If vector\_store is set, returns None after storing all chunks. |

---


## Modules

<a id="33"></a>

## Modules - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/

# Modules

This section contains API reference documentation for all datapizza-ai modules. Modules are organized by functionality and include both core modules (included by default) and optional modules that require separate installation.

## Core Modules (Included by Default)

These modules are included with `datapizza-ai-core` and are available without additional installation:

* [Parsers](Parsers/) - Convert documents into structured Node representations
* [Captioners](captioners/) - Generate captions and descriptions for content
* [Metatagger](metatagger/) - Add metadata tags to content
* [Prompt](Prompt/ChatPromptTemplate/) - Manage prompts and prompt templates
* [Rewriters](rewriters/) - Transform and rewrite content
* [Splitters](Splitters/) - Split content into smaller chunks
* [Treebuilder](treebuilder/) - Build hierarchical tree structures from content

## Optional Modules (Separate Installation Required)

These modules require separate installation via pip:

* [Rerankers](Rerankers/) - Rerank and score content relevance

Each module page includes installation instructions and usage examples.

---


### Parsers

<a id="35"></a>

## Parsers - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Parsers/

# Parsers

Parsers are pipeline components that convert documents into structured hierarchical Node representations. They extract text, layout information, and metadata from various document formats to create tree-like data structures for further processing.

Each parser should return a [Node](../../Type/node/) object, which is a hierarchical representation of the document content.

If you write a custom parser that returns a different type of object (for example, the plain text of the document content), you must use a [TreeBuilder](../treebuilder/) to convert it into a Node.

## Available Parsers

### Core Parsers (Included by Default)

* [TextParser](text_parser/) - Simple text parser for plain text content

### Optional Parsers (Separate Installation Required)

* [AzureParser](azure_parser/) - Azure AI Document Intelligence parser for PDFs and documents
* [DoclingParser](docling_parser/) - Docling-based parser for PDFs with layout preservation and media extraction

## Common Usage Patterns

### Basic Text Processing

```
from datapizza.modules.parsers.text_parser import parse_text

# Process plain text
document = parse_text("Your text content here")
```

### Document Processing Pipeline

```
from datapizza.modules.parsers import TextParser
from datapizza.modules.splitters import RecursiveSplitter

# Create processing pipeline
parser = TextParser()
splitter = RecursiveSplitter(chunk_size=1000)

# Process document
document = parser.parse(text_content)
chunks = splitter(document.content)
```

---

<a id="36"></a>

## TextParser - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Parsers/text_parser/

# TextParser

## datapizza.modules.parsers.TextParser

Bases: `Parser`

Parser that creates a hierarchical tree structure from text.
The hierarchy goes from document -> paragraphs -> sentences.

### \_\_init\_\_

```
__init__()
```

Initialize the TextParser.

### parse

```
parse(text, metadata=None)
```

Parse text into a hierarchical tree structure.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` | `str` | The text to parse | *required* |
| `metadata` | `dict | None` | Optional metadata for the root node | `None` |

Returns:

| Type | Description |
| --- | --- |
| `Node` | A Node representing the document with paragraph and sentence nodes |

## Usage

```
from datapizza.modules.parsers.text_parser import TextParser, parse_text

# Using the class
parser = TextParser()
document_node = parser.parse("Your text content here", metadata={"source": "example"})

# Using the convenience function
document_node = parse_text("Your text content here")
```

## Parameters

The TextParser class takes no initialization parameters.

The `parse` method accepts:
- `text` (str): The text content to parse
- `metadata` (dict, optional): Additional metadata to attach to the document

## Features

* Splits text into paragraphs based on double newlines
* Breaks paragraphs into sentences using regex patterns
* Creates three-level hierarchy: document → paragraphs → sentences
* Preserves original text content in sentence nodes
* Adds index metadata for paragraphs and sentences

## Node Types Created

* `DOCUMENT`: Root document container
* `PARAGRAPH`: Text paragraphs
* `SENTENCE`: Individual sentences with content

## Examples

### Basic Usage

```
from datapizza.modules.parsers.text_parser import parse_text

text_content = """
This is the first paragraph.
It contains multiple sentences.

This is the second paragraph.
It also has content.
"""

document = parse_text(text_content, metadata={"source": "user_input"})

# Navigate structure
for i, paragraph in enumerate(document.children):
    print(f"Paragraph {i}:")
    for j, sentence in enumerate(paragraph.children):
        print(f"  Sentence {j}: {sentence.content}")
```

### Class-Based Usage

```
from datapizza.modules.parsers.text_parser import TextParser

parser = TextParser()

# Parse with custom metadata
document = parser.parse(
    text="Sample text content here.",
    metadata={
        "source": "api_input",
        "timestamp": "2024-01-01",
        "language": "en"
    }
)

# Access document metadata
print(f"Source: {document.metadata['source']}")
print(f"Number of paragraphs: {len(document.children)}")
```

### Pipeline Integration

```
from datapizza.modules.parsers.text_parser import TextParser
from datapizza.modules.splitters import RecursiveSplitter

# Create processing pipeline
parser = TextParser()
splitter = RecursiveSplitter(chunk_size=500)

def process_text_document(text):
    # Parse into hierarchical structure
    document = parser.parse(text)

    # Convert back to flat text for splitting
    full_text = document.content

    # Split into chunks
    chunks = splitter(full_text)

    return document, chunks

# Process document
structured_doc, chunks = process_text_document(long_text)
```

## Best Practices

1. **Use for Simple Text**: Best suited for plain text content without complex formatting
2. **Preprocessing**: Clean text of unwanted characters before parsing if needed
3. **Metadata**: Add relevant metadata during parsing for downstream processing
4. **Pipeline Integration**: Combine with other modules for complete text processing workflows

---

<a id="37"></a>

## DoclingParser - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Parsers/docling_parser/

# DoclingParser

A document parser that uses Docling to convert PDF files into structured hierarchical Node representations with preserved layout information and media extraction.

## Installation

```
pip install datapizza-ai-parsers-docling
```

## Usage

```
from datapizza.modules.parsers.docling import DoclingParser

# Basic usage
parser = DoclingParser()
document_node = parser.parse("sample.pdf")

print(document_node)
```

## Parameters

* `json_output_dir` (str, optional): Directory to save intermediate Docling JSON results for debugging and inspection

## Features

* **PDF Processing**: Converts PDF files using Docling's DocumentConverter with OCR and table structure detection
* **Hierarchical Structure**: Creates logical document hierarchy (document → sections → paragraphs/tables/figures)
* **Media Extraction**: Extracts images and tables as base64-encoded media with bounding box coordinates
* **Layout Preservation**: Maintains spatial layout information including page numbers and bounding regions
* **Markdown Generation**: Converts tables to markdown format and handles list structures
* **Metadata Rich**: Preserves full Docling metadata in `docling_raw` with convenience fields

## Configuration

The parser automatically configures Docling with:

* Table structure detection enabled
* Full page OCR with EasyOCR
* PyPdfium backend for PDF processing

## Examples

### Basic Document Processing

```
from datapizza.modules.parsers.docling import DoclingParser

parser = DoclingParser()
document = parser.parse("research_paper.pdf")

# Access hierarchical structure
for section in document.children:
    print(f"Section: {section.metadata.get('docling_label', 'Unknown')}")
    for child in section.children:
        if child.node_type.name == "PARAGRAPH":
            print(f"  Paragraph: {child.content[:100]}...")
        elif child.node_type.name == "TABLE":
            print(f"  Table with {len(child.children)} rows")
        elif child.node_type.name == "FIGURE":
            print(f"  Figure: {child.metadata.get('docling_label', 'Image')}")
```

### Configured OCR Document Processing

```
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.parsers.docling.ocr_options import OCROptions, OCREngine

# Configure parser with EasyOCR (default, backward compatible)
parser = DoclingParser()
document = parser.parse("document.pdf")

# Configure parser with Tesseract OCR for Italian language
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["ita"],
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("italian_document.pdf")

# Configure parser with Tesseract for multiple languages
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["eng", "fra"],  # English and French
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("multilingual_document.pdf")

# Configure parser with Tesseract for Italian and English
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["ita", "eng"],
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("italian_english_document.pdf")

# Configure parser with automatic language detection
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["auto"],
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("mixed_language_document.pdf")

# Disable OCR completely (for documents without text that needs OCR)
ocr_config = OCROptions(engine=OCREngine.NONE)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("native_text_document.pdf")

# Enable custom EasyOCR configuration
ocr_config = OCROptions(
    engine=OCREngine.EASY_OCR,
    easy_ocr_force_full_page=False,  # Process only text-light regions
)
parser = DoclingParser(ocr_options=ocr_config)
document = parser.parse("document.pdf")

# Parse with JSON output for debugging
ocr_config = OCROptions(engine=OCREngine.TESSERACT, tesseract_lang=["ita", "eng"])
parser = DoclingParser(
    json_output_dir="./docling_debug",
    ocr_options=ocr_config,
)
document = parser.parse("document.pdf")
# Intermediate Docling JSON will be saved to ./docling_debug/document.json
```

### Tesseract Language Options

When using Tesseract OCR, you can specify languages in the `tesseract_lang` parameter as a list:

**Single Language:**

```
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["eng"],  # English
)
```

**Multiple Languages:**

```
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["eng", "ita", "fra"],  # English, Italian, French
)
```

**Automatic Language Detection:**

```
ocr_config = OCROptions(
    engine=OCREngine.TESSERACT,
    tesseract_lang=["auto"],  # Auto-detect language
)
```

**Common Language Codes:**
- `"eng"` - English
- `"ita"` - Italian
- `"fra"` - French
- `"deu"` - German
- `"spa"` - Spanish
- `"por"` - Portuguese
- `"chi_sim"` - Simplified Chinese
- `"chi_tra"` - Traditional Chinese
- `"jpn"` - Japanese
- `"auto"` - Automatic detection

For a complete list of supported languages, refer to [Tesseract documentation](https://github.com/UB-Mannheim/tesseract/wiki).

---

<a id="38"></a>

## AzureParser - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Parsers/azure_parser/

# AzureParser

A document parser that uses Azure AI Document Intelligence to extract structured content from PDFs and other documents.

## Installation

```
pip install datapizza-ai-parsers-azure
```

## datapizza.modules.parsers.azure.AzureParser

Bases: `Parser`

Parser that creates a hierarchical tree structure from Azure AI Document Intelligence response.
The hierarchy goes from document -> pages -> paragraphs/tables -> lines/cells -> words.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str` | str | *required* |
| `endpoint` | `str` | str | *required* |
| `result_type` | `str` | str = "markdown", "text" | `'text'` |

### \_\_call\_\_

```
__call__(file_path, metadata=None)
```

Allow the parser to be called directly as a function.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file_path` | `str` | Path to the document | *required* |
| `metadata` | `dict | None` | Optional metadata to be merged into the root document node | `None` |

Returns:

| Type | Description |
| --- | --- |
| `Node` | A Node representing the document with hierarchical structure |

### a\_parse `async`

```
a_parse(file_path, metadata=None)
```

Async version of parse().

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file_path` | `str` | Path to the document | *required* |
| `metadata` | `dict | None` | Optional metadata to be merged into the root document node. Defaults to None. | `None` |

Returns:

| Type | Description |
| --- | --- |
| `Node` | A Node representing the document with hierarchical structure |

Raises:

| Type | Description |
| --- | --- |
| `TypeError` | If metadata is not a dict or None |

### parse

```
parse(file_path, metadata=None)
```

Parse a Document with Azure AI Document Intelligence into a Node
structure.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file_path` | `str` | Path to the document | *required* |
| `metadata` | `dict | None` | Optional metadata to be merged into the root document node. Defaults to None. | `None` |

Returns:

| Type | Description |
| --- | --- |
| `Node` | A Node representing the document with hierarchical structure |

Raises:

| Type | Description |
| --- | --- |
| `TypeError` | If metadata is not a dict or None |

### parse\_with\_azure\_ai

```
parse_with_azure_ai(file_path)
```

Parse a Document with Azure AI Document Intelligence into a json dictionary.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file_path` | `str` | Path to the document | *required* |

Returns:

| Type | Description |
| --- | --- |
| `dict` | A dictionary with the Azure AI Document Intelligence response |

## Usage

```
from datapizza.modules.parsers.azure import AzureParser

parser = AzureParser(
    api_key="your-azure-key",
    endpoint="https://your-endpoint.cognitiveservices.azure.com/",
    result_type="text"
)

document_node = parser.parse("document.pdf")
```

## Parameters

* `api_key` (str): Azure AI Document Intelligence API key
* `endpoint` (str): Azure service endpoint URL
* `result_type` (str): Output format - "text" or "markdown" (default: "text")

## Features

* Creates hierarchical document structure: document → sections → paragraphs/tables/figures
* Extracts bounding regions and spatial layout information
* Handles tables, figures, and complex document layouts
* Preserves metadata including page numbers and coordinates
* Supports both sync and async processing
* Converts media elements to base64 images with coordinates

## Node Types Created

* `DOCUMENT`: Root document container
* `SECTION`: Document sections
* `PARAGRAPH`: Text paragraphs with content
* `TABLE`: Tables with markdown representation
* `FIGURE`: Images and figures with media data

## Examples

### Basic Document Processing

```
from datapizza.modules.parsers.azure import AzureParser
import os

parser = AzureParser(
    api_key=os.getenv("AZURE_DOC_INTELLIGENCE_KEY"),
    endpoint=os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT"),
    result_type="markdown"
)

# Parse document
document = parser.parse("complex_document.pdf")

# Access hierarchical structure
for section in document.children:
    for paragraph in section.children:
        print(f"Content: {paragraph.content}")
        print(f"Bounding regions: {paragraph.metadata.get('boundingRegions', [])}")
```

### Async Processing

```
async def process_document():
    document = await parser.a_run("document.pdf")
    return document

# Usage in async context
document = await process_document()
```

---


### Treebuilder

<a id="50"></a>

## Treebuilder - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/treebuilder/

# Treebuilder

Treebuilders are pipeline components that construct hierarchical tree structures (Node objects) from various types of content. They convert flat or unstructured content into organized, nested representations that facilitate better processing and understanding.

## datapizza.modules.treebuilder.LLMTreeBuilder

TreeBuilder that creates a hierarchical tree structure from text input using an LLM.
The hierarchy goes from document -> sections -> paragraphs -> sentences.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `client` | `Client` | Client - An instance of an LLM client (e.g., GeminiClient) | *required* |

### invoke

```
invoke(file_path)
```

Invoke the tree builder on the input text using an LLM.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file_path` | `str` | Path to the file to process | *required* |

Returns:

| Type | Description |
| --- | --- |
| `Node` | A Node representing the document with hierarchical structure |

### parse

```
parse(text)
```

Build a tree from the input text using an LLM.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` | `str` | Input text to process | *required* |

Returns:

| Type | Description |
| --- | --- |
| `Node` | A Node representing the document with hierarchical structure |

A treebuilder that uses language models to analyze content and create hierarchical structures based on semantic understanding.

```
from datapizza.clients.openai import OpenAIClient
from datapizza.modules.treebuilder import LLMTreeBuilder

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o")
treebuilder = LLMTreeBuilder(client=client)

flat_content = "This is a flat piece of content. It should be converted into a hierarchical structure."

structured_document = treebuilder.parse(flat_content)

print(structured_document)
```

**Features:**

* Semantic understanding of content organization
* Configurable tree depth and structure rules
* Support for various content types (articles, reports, manuals, etc.)
* Preserves original content while adding hierarchical organization
* Metadata extraction and tagging during structure creation
* Supports both sync and async processing

## Usage Examples

### Basic Tree Structure Creation

```
from datapizza.modules.treebuilder import LLMTreeBuilder
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(api_key="your-openai-key")

# Create basic treebuilder
treebuilder = LLMTreeBuilder(client=client)

# Unstructured content
flat_content = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

Supervised Learning
In supervised learning, algorithms learn from labeled training data. The goal is to predict outcomes for new data based on patterns learned from the training set. Common examples include classification and regression tasks.

Classification algorithms predict discrete categories or classes. For example, email spam detection classifies emails as either spam or not spam.

Regression algorithms predict continuous numerical values. For instance, predicting house prices based on features like location and size.

Unsupervised Learning
Unsupervised learning finds patterns in data without labeled examples. The algorithm identifies hidden structures in input data.

Clustering groups similar data points together. Customer segmentation is a common application.

Dimensionality reduction reduces the number of features while preserving important information.

Reinforcement Learning
This approach learns through interaction with an environment, receiving rewards or penalties for actions taken.
"""

# Build hierarchical structure
structured_document = treebuilder.parse(flat_content)

# Navigate the structure
def print_structure(node, depth=0):
    indent = "  " * depth
    print(f"{indent}{node.node_type.value}: {node.content[:50]}...")
    for child in node.children:
        print_structure(child, depth + 1)

print_structure(structured_document)
```

---


### Captioners

<a id="34"></a>

## Captioners - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/captioners/

# Captioners

Captioners are pipeline components that generate captions and descriptions for media content such as images, figures, and tables. They use LLM clients to analyze visual content and produce descriptive text.

### LLMCaptioner

## datapizza.modules.captioners.LLMCaptioner

Bases: `NodeCaptioner`

Captioner that uses an LLM client to caption a node.

### \_\_init\_\_

```
__init__(
    client,
    max_workers=3,
    system_prompt_table="Generate concise captions for tables.",
    system_prompt_figure="Generate descriptive captions for figures.",
)
```

Captioner that uses an LLM client to caption a node.
Args:
client: The LLM client to use.
max\_workers: The maximum number of workers to use. in sync mode is the number of threads spawned, in async mode is the number of batches.
system\_prompt\_table: The system prompt to use for table captioning.
system\_prompt\_figure: The system prompt to use for figure captioning.

### a\_caption `async`

```
a_caption(node)
```

async Caption a node.
Args:
node: The node to caption.

Returns:

| Type | Description |
| --- | --- |
| `Node` | The same node with the caption. |

### a\_caption\_media `async`

```
a_caption_media(media, system_prompt=None)
```

async Caption image.
Args:
media: The media to caption.
system\_prompt: Optional system prompt to guide the captioning.

Returns:

| Type | Description |
| --- | --- |
| `str` | The string caption. |

### caption

```
caption(node)
```

Caption a node.
Args:
node: The node to caption.

Returns:

| Type | Description |
| --- | --- |
| `Node` | The same node with the caption. |

### caption\_media

```
caption_media(media, system_prompt=None)
```

Caption an image.
Args:
media: The media to caption.
system\_prompt: Optional system prompt to guide the captioning.

Returns:

| Type | Description |
| --- | --- |
| `str` | The string caption. |

A captioner that uses language models to generate captions for media nodes (figures and tables) within document hierarchies.

```
from datapizza.clients.openai import OpenAIClient
from datapizza.modules.captioners import LLMCaptioner
from datapizza.type import ROLE, Media, MediaNode, NodeType

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o")
captioner = LLMCaptioner(
    client=client,
    max_workers=3,
    system_prompt_table="Describe this table in detail.",
    system_prompt_figure="Describe this figure/image in detail."
)

document_node = MediaNode( node_type=NodeType.FIGURE, children=[], metadata={}, media=Media(source_type="path", source="gogole.png", extension="png", media_type="image"))
captioned_document = captioner(document_node)
print(captioned_document)
```

**Parameters:**

* `client` (Client): The LLM client to use for caption generation
* `max_workers` (int): Maximum number of concurrent workers for parallel processing (default: 3)
* `system_prompt_table` (str, optional): System prompt for table captioning
* `system_prompt_figure` (str, optional): System prompt for figure captioning

**Features:**

* Automatically finds all media nodes (figures and tables) in a document hierarchy
* Generates captions using configurable system prompts
* Supports concurrent processing for better performance
* Creates new paragraph nodes containing the original content plus generated captions
* Preserves original node metadata and structure
* Supports both sync and async processing

**Supported Node Types:**

* `FIGURE`: Images and visual figures
* `TABLE`: Tables and tabular data

**Output Format:**

The captioner creates new paragraph nodes with content in the format:

```
{original_content} <{node_type}> [{generated_caption}]
```

---


### Splitters

<a id="44"></a>

## Splitters - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Splitters/

# Splitters

Splitters are pipeline components that divide large text content into smaller, manageable chunks. They help optimize content for processing, storage, and retrieval in AI applications by creating appropriately sized segments while preserving context and meaning.

## Installation

All splitters are included with `datapizza-ai-core` and require no additional installation.

## Available Splitters

### Core Splitters (Included by Default)

* [RecursiveSplitter](recursive_splitter/) - Recursively divides text using multiple splitting strategies
* [TextSplitter](text_splitter/) - Basic text splitter for general-purpose chunking
* [NodeSplitter](node_splitter/) - Splitter for Node objects preserving hierarchical structure
* [PDFImageSplitter](pdf_image_splitter/) - Specialized splitter for PDF content with images

## Common Features

* Multiple splitting strategies for different content types
* Configurable chunk sizes and overlap
* Context preservation through overlapping
* Support for structured content (nodes, PDFs, etc.)
* Metadata preservation during splitting
* Spatial layout awareness for document content

## Usage Patterns

### Basic Text Splitting

```
from datapizza.modules.splitters import RecursiveSplitter

splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter(long_text_content)
```

### Document Processing Pipeline

```
from datapizza.modules.parsers import TextParser
from datapizza.modules.splitters import NodeSplitter

parser = TextParser()
splitter = NodeSplitter(max_char = 4000)

document = parser.parse(text_content)
structured_chunks = splitter(document)
```

### Choosing the Right Splitter

* **RecursiveSplitter**: Best for general text content, articles, and most use cases
* **TextSplitter**: Simple splitting for basic text without complex requirements
* **NodeSplitter**: When working with structured Node objects from parsers
* **PDFImageSplitter**: Specifically for PDF content with images and complex layouts
* **BBoxMerger**: Utility for processing documents with spatial layout information

---

<a id="47"></a>

## RecursiveSplitter - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Splitters/recursive_splitter/

# RecursiveSplitter

## datapizza.modules.splitters.RecursiveSplitter

Bases: `Splitter`

The RecursiveSplitter takes leaf nodes from a tree document structure and groups them into Chunk objects until reaching the maximum character limit. Each leaf Node represents the smallest unit of content that can be grouped.

### \_\_init\_\_

```
__init__(max_char=5000, overlap=0)
```

Initialize the RecursiveSplitter.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `max_char` | `int` | The maximum number of characters per chunk | `5000` |
| `overlap` | `int` | The number of characters to overlap between chunks | `0` |

### split

```
split(node)
```

Split the node into chunks.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `node` | `Node` | The node to split | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | A list of chunks |

## Usage

```
from datapizza.modules.parsers import TextParser
from datapizza.modules.splitters import RecursiveSplitter

splitter = RecursiveSplitter(
    max_char=10,
    overlap=1,
)

# Parse text into nodes because RecursiveSplitter need Node
parser = TextParser()
document = parser.parse("""
This is the first section of the document.
It contains important information about the topic.

This is the second section with more details.
It provides additional context and examples.

The final section concludes the document.
It summarizes the key points discussed.
""")

chunks = splitter.split(document)
print(chunks)
```

## Features

* Uses multiple separator strategies in order of preference
* Recursive approach ensures optimal chunk boundaries
* Configurable chunk size and overlap for context preservation
* Handles various content types with appropriate separator selection
* Preserves content structure while maintaining size limits

---

<a id="48"></a>

## TextSplitter - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Splitters/text_splitter/

# TextSplitter

## datapizza.modules.splitters.TextSplitter

Bases: `Splitter`

A basic text splitter that operates directly on strings rather than Node objects.
Unlike other splitters that work with Node types, this splitter takes raw text input
and splits it into chunks while maintaining configurable size and overlap parameters.

### \_\_init\_\_

```
__init__(max_char=5000, overlap=0)
```

Initialize the TextSplitter.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `max_char` | `int` | The maximum number of characters per chunk | `5000` |
| `overlap` | `int` | The number of characters to overlap between chunks | `0` |

### split

```
split(text)
```

Split the text into chunks.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` | `str` | The text to split | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | A list of chunks |

## Usage

```
from datapizza.modules.splitters import TextSplitter

splitter = TextSplitter(
    max_char=500,
    overlap=50
)

chunks = splitter.split(text_content)
```

## Features

* Simple, straightforward text splitting algorithm
* Configurable chunk size and overlap
* Lightweight implementation for basic splitting needs
* Preserves character-level accuracy in chunk boundaries
* Minimal overhead for high-performance applications

## Examples

### Basic Usage

```
from datapizza.modules.splitters import TextSplitter

splitter = TextSplitter(max_char=50, overlap=5)

text = """
This is a sample text that we want to split into smaller chunks.
The TextSplitter will divide this content based on the specified
chunk size and overlap parameters. This ensures that information
is preserved while creating manageable pieces of content.
"""

chunks = splitter.split(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.text)} chars")
    print(f"Content: {chunk.text}")
    print("---")
```

---

<a id="45"></a>

## NodeSplitter - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Splitters/node_splitter/

# NodeSplitter

## datapizza.modules.splitters.NodeSplitter

Bases: `Splitter`

A splitter that traverses a document tree from the root node. If the root node's content is smaller than max\_chars,
it becomes a single chunk. Otherwise, it recursively processes the node's children, creating chunks from the first
level of children that fit within max\_chars, continuing deeper into the tree structure as needed.

### \_\_init\_\_

```
__init__(max_char=5000)
```

Initialize the NodeSplitter.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `max_char` | `int` | The maximum number of characters per chunk | `5000` |

### split

```
split(node)
```

Split the node into chunks.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `node` | `Node` | The node to split | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | A list of chunks |

## Usage

```
from datapizza.modules.splitters import NodeSplitter

splitter = NodeSplitter(
    max_char=800,
)

node_chunks = splitter.split(document_node)
```

## Features

* Maintains Node object structure and hierarchy
* Preserves metadata from original nodes
* Respects node boundaries when possible
* Supports both structure-preserving and flattened chunking
* Handles nested node relationships intelligently

## Examples

### Basic Node Splitting

```
from datapizza.modules.parsers import TextParser
from datapizza.modules.splitters import NodeSplitter

# Parse text into nodes
parser = TextParser()
document = parser.parse("""
This is the first section of the document.
It contains important information about the topic.

This is the second section with more details.
It provides additional context and examples.

The final section concludes the document.
It summarizes the key points discussed.
""")

splitter = NodeSplitter(
    max_char=150,
)

chunks = splitter.split(document)

# Examine the structured chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Content length: {len(chunk.text)}")
    print(f"  Content preview: {chunk.text[:80]}...")
    print("---")
```

---

<a id="46"></a>

## PDFImageSplitter - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Splitters/pdf_image_splitter/

# PDFImageSplitter

## datapizza.modules.splitters.PDFImageSplitter

Bases: `Splitter`

Splits a PDF document into individual pages, saves each page as an image using fitz,
and returns metadata about each page as a Chunk object.

### \_\_init\_\_

```
__init__(
    image_format="png",
    output_base_dir="output_images",
    dpi=300,
)
```

Initializes the Splitter.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `image_format` | `Literal['png', 'jpeg']` | The format to save the images in ('png' or 'jpeg'). Defaults to 'png'. | `'png'` |
| `output_base_dir` | `str | Path` | The base directory where images for processed PDFs will be saved. A subdirectory will be created for each PDF. Defaults to 'output\_images'. | `'output_images'` |
| `dpi` | `int` | Dots Per Inch for rendering the PDF page to an image. Higher values increase resolution and file size. Defaults to 300. | `300` |

### split

```
split(pdf_path)
```

Processes the PDF using fitz: converts pages to images and returns Chunk objects.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pdf_path` | `str | Path` | The path to the input PDF file. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | A list of Chunk objects, one for each page of the PDF. |

## Usage

```
from datapizza.modules.splitters import PDFImageSplitter

splitter = PDFImageSplitter()

pdf_chunks = splitter("pdf_path")
```

## Features

* Specialized handling of PDF document structure
* Preserves image data and visual elements
* Maintains spatial layout information
* Includes page-level metadata and coordinates
* Handles complex document layouts with mixed content
* Optimized for PDF content from document intelligence services

## Examples

### Basic PDF Content Splitting

```
from datapizza.modules.splitters import PDFImageSplitter

# Split while preserving images and layout
pdf_splitter = PDFImageSplitter()

pdf_chunks = pdf_splitter("pdf_path")

# Examine chunks with visual content
for i, chunk in enumerate(pdf_chunks):
    print(f"Chunk {i+1}:")
    print(f"  Content length: {len(chunk.content)}")
    print(f"  Page: {chunk.metadata.get('page_number', 'unknown')}")

    if hasattr(chunk, 'media') and chunk.media:
        print(f"  Media elements: {len(chunk.media)}")
        for media in chunk.media:
            print(f"    Type: {media.media_type}")

    if 'boundingRegions' in chunk.metadata:
        print(f"  Bounding regions: {len(chunk.metadata['boundingRegions'])}")

    print("---")
```

---


### Metatagger

<a id="49"></a>

## Metatagger - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/metatagger/

# Metatagger

Metataggers are pipeline components that add metadata tags to content chunks using language models. They analyze text content and generate relevant keywords, tags, or other metadata to enhance content discoverability and organization.

## datapizza.modules.metatagger.KeywordMetatagger

Bases: `Metatagger`

Keyword metatagger that uses an LLM client to add metadata to a chunk.

### \_\_init\_\_

```
__init__(
    client,
    max_workers=3,
    system_prompt=None,
    user_prompt=None,
    keyword_name="keywords",
)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `client` | `Client` | The LLM client to use. | *required* |
| `max_workers` | `int` | The maximum number of workers to use. | `3` |
| `system_prompt` | `str | None` | The system prompt to use. | `None` |
| `user_prompt` | `str | None` | The user prompt to use. | `None` |
| `keyword_name` | `str` | The name of the keyword field. | `'keywords'` |

### a\_tag `async`

```
a_tag(chunks)
```

async Add metadata to a chunk.

### tag

```
tag(chunks)
```

Add metadata to a chunk.

A metatagger that uses language models to generate keywords and metadata for text chunks.

```
from datapizza.modules.metatagger import KeywordMetatagger
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(api_key="your-api-key")
metatagger = KeywordMetatagger(
    client=client,
    max_workers=3,
    system_prompt="Generate relevant keywords for the given text.",
    user_prompt="Extract 5-10 keywords from this text:",
    keyword_name="keywords"
)

# Process chunks
tagged_chunks = metatagger.tag(chunks)
```

**Features:**

* Processes chunks in parallel for better performance
* Configurable prompts for different keyword extraction strategies
* Adds generated keywords to chunk metadata
* Supports custom metadata field naming
* Handles both individual chunks and lists of chunks
* Uses memory-based conversation for consistent prompting

**Input/Output:**

* Input: `Chunk` objects or lists of `Chunk` objects
* Output: Same `Chunk` objects with additional metadata containing generated keywords

## Usage Examples

### Basic Keyword Extraction

```
import uuid

from datapizza.clients.openai import OpenAIClient
from datapizza.modules.metatagger import KeywordMetatagger
from datapizza.type import Chunk

# Initialize client and metatagger
client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o")
metatagger = KeywordMetatagger(
    client=client,
    system_prompt="You are a keyword extraction expert. Generate relevant, concise keywords.",
    user_prompt="Extract 5-8 important keywords from this text:",
    keyword_name="keywords"
)

# Process chunks
chunks = [
    Chunk(id=str(uuid.uuid4()), text="Machine learning algorithms are transforming healthcare diagnostics."),
    Chunk(id=str(uuid.uuid4()), text="Climate change impacts ocean temperatures and marine ecosystems.")
]

tagged_chunks = metatagger.tag(chunks)

# Access generated keywords
for chunk in tagged_chunks:
    print(f"Content: {chunk.text}")
    print(f"Keywords: {chunk.metadata.get('keywords', [])}")
```

---


### Rewriters

<a id="43"></a>

## Rewriters - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/rewriters/

# Rewriters

Rewriters are pipeline components that transform and rewrite text content using language models. They can modify content style, format, tone, or structure while preserving meaning and important information.

## datapizza.modules.rewriters.ToolRewriter

Bases: `Rewriter`

A tool-based query rewriter that uses LLMs to transform user queries through structured tool interactions.

### a\_rewrite `async`

```
a_rewrite(user_prompt, memory=None)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `user_prompt` | `str` | The user query to rewrite. | *required* |
| `memory` | `Memory | None` | The memory to use for the rewrite. | `None` |

Returns:

| Type | Description |
| --- | --- |
| `str` | The rewritten query. |

### rewrite

```
rewrite(user_prompt, memory=None)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `user_prompt` | `str` | The user query to rewrite. | *required* |
| `memory` | `Memory | None` | The memory to use for the rewrite. | `None` |

Returns:

| Type | Description |
| --- | --- |
| `str` | The rewritten query. |

A rewriter that uses language models to transform text content with specific instructions and tools.

```
from datapizza.clients.openai import OpenAIClient
from datapizza.modules.rewriters import ToolRewriter

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o")

# Create a simplification rewriter
simplifier = ToolRewriter(
    client=client,
    system_prompt="You are an expert at simplifying complex text for general audiences.",
)

# Simplify technical content
technical_text = """
The algorithmic implementation utilizes a recursive binary search methodology
to optimize computational complexity in logarithmic time scenarios.
"""

simplified_text = simplifier(technical_text)
print(simplified_text)
# Output: recursive binary search
```

**Features:**

* Flexible content transformation with custom instructions
* Support for various rewriting tasks (summarization, style changes, format conversion)
* Integration with tool calling for enhanced capabilities
* Preserves important information while transforming presentation
* Supports both sync and async processing
* Configurable prompting for different rewriting strategies

---


### Rerankers

<a id="40"></a>

## Rerankers - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Rerankers/

# Rerankers

Rerankers are pipeline components that reorder and score retrieved content based on relevance to a query. They improve retrieval quality by applying more sophisticated ranking algorithms after initial retrieval, helping surface the most relevant content for user queries.

## Installation

All rerankers require separate installation via pip and are not included by default with `datapizza-ai-core`.

## Available Rerankers

### Optional Rerankers (Separate Installation Required)

* [CohereReranker](cohere_reranker/) - Uses Cohere's reranking API for high-quality semantic reranking
* [TogetherReranker](together_reranker/) - Uses Together AI's API with various model options

## Common Features

* High-quality semantic reranking using specialized models
* Configurable result count and score thresholds
* Support for both sync and async processing
* Automatic relevance scoring for retrieved content
* Integration with various reranking model providers

## Usage Patterns

### Basic Reranking Pipeline

```
from datapizza.modules.rerankers.cohere import CohereReranker

reranker = CohereReranker(
    api_key="your-cohere-key",
    endpoint="https://api.cohere.ai/v1",
    top_n=5,
    threshold=0.6
)

query = "What is deep learning?"
reranked_chunks = reranker(query, chunks)
```

### RAG Pipeline Integration

```
from datapizza.modules.rerankers.together import TogetherReranker
from datapizza.vectorstores import QdrantVectorStore

# Initial broad retrieval
vectorstore = QdrantVectorStore(collection_name="documents")
initial_results = vectorstore.similarity_search(query, k=20)

# Rerank for better relevance
reranker = TogetherReranker(api_key="together-key", model="rerank-model")
reranked_results = reranker(query, initial_results)
```

## Best Practices

1. **Choose the Right Model**: Select reranker models based on your domain and language requirements
2. **Tune Thresholds**: Experiment with relevance score thresholds to balance precision and recall
3. **Initial Retrieval Size**: Retrieve more documents initially (k=20-50) before reranking to improve final quality
4. **Performance Considerations**: Use async processing for high-throughput applications
5. **Cost Management**: Monitor API usage, especially for high-volume applications
6. **Evaluation**: Test different rerankers on your specific data to find the best performance

---

<a id="41"></a>

## CohereReranker - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Rerankers/cohere_reranker/

# CohereReranker

A reranker that uses Cohere's reranking API to score and reorder documents based on query relevance.

## Installation

```
pip install datapizza-ai-rerankers-cohere
```

## datapizza.modules.rerankers.cohere.CohereReranker

Bases: `Reranker`

A reranker that uses the Cohere API to rerank documents.

### \_\_init\_\_

```
__init__(
    api_key,
    endpoint,
    top_n=10,
    threshold=None,
    model="model",
)
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str` | The API key for the Cohere API. | *required* |
| `endpoint` | `str` | The endpoint for the Cohere API. | *required* |
| `top_n` | `int` | The number of documents to return. | `10` |
| `threshold` | `float | None` | The threshold for the reranker. | `None` |

### a\_rerank `async`

```
a_rerank(query, documents)
```

Rerank documents based on query.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `query` | `str` | The query to rerank documents by. | *required* |
| `documents` | `list[Chunk]` | The documents to rerank. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | The reranked documents. |

### rerank

```
rerank(query, documents)
```

Rerank documents based on query.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `query` | `str` | The query to rerank documents by. | *required* |
| `documents` | `list[Chunk]` | The documents to rerank. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | The reranked documents. |

## Usage

```
from datapizza.modules.rerankers.cohere import CohereReranker

reranker = CohereReranker(
    api_key="your-cohere-api-key",
    endpoint="https://api.cohere.ai/v1",
    top_n=10,
    threshold=0.5,
    model="rerank-v3.5",
)

# Rerank chunks based on query
query = "What are the benefits of machine learning?"
reranked_chunks = reranker.rerank(query, chunks)
```

## Features

* High-quality semantic reranking using Cohere's models
* Configurable result count and score thresholds
* Support for both sync and async processing
* Automatic relevance scoring for retrieved content
* Integration with Cohere's latest reranking models
* Flexible endpoint configuration for different Cohere services

## Examples

### Basic Usage

```
import uuid

from datapizza.modules.rerankers.cohere import CohereReranker
from datapizza.type import Chunk

# Initialize reranker
reranker = CohereReranker(
    api_key="COHERE_API_KEY",
    endpoint="https://api.cohere.ai/v1",
    top_n=5,
    threshold=0.6,
    model="rerank-v3.5",
)

# Sample retrieved chunks
chunks = [
    Chunk(id=str(uuid.uuid4()), text="Machine learning enables computers to learn from data..."),
    Chunk(id=str(uuid.uuid4()), text="Deep learning is a subset of machine learning..."),
    Chunk(id=str(uuid.uuid4()), text="Neural networks consist of interconnected nodes..."),
    Chunk(id=str(uuid.uuid4()), text="Supervised learning uses labeled training data..."),
    Chunk(id=str(uuid.uuid4()), text="The weather forecast shows rain tomorrow...")
]

query = "What is deep learning and how does it work?"

# Rerank based on relevance to query
reranked_chunks = reranker.rerank(query, chunks)

# Display results with scores
for i, chunk in enumerate(reranked_chunks):
    score = chunk.metadata.get('relevance_score', 'N/A')
    print(f"Rank {i+1} (Score: {score}): {chunk.text[:80]}...")
```

---

<a id="42"></a>

## TogetherReranker - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Rerankers/together_reranker/

# TogetherReranker

A reranker that uses Together AI's API for document reranking with various model options.

## Installation

```
pip install datapizza-ai-rerankers-together
```

## datapizza.modules.rerankers.together.TogetherReranker

Bases: `Reranker`

A reranker that uses the Together API to rerank documents.

### \_\_init\_\_

```
__init__(api_key, model, top_n=10, threshold=None)
```

Initialize the TogetherReranker.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str` | Together API key | *required* |
| `model` | `str` | Model name to use for reranking | *required* |
| `top_n` | `int` | Number of top documents to return | `10` |
| `threshold` | `Optional[float]` | Minimum relevance score threshold. If None, no filtering is applied. | `None` |

### rerank

```
rerank(query, documents)
```

Rerank documents based on query.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `query` | `str` | The query to rerank documents by. | *required* |
| `documents` | `list[Chunk]` | The documents to rerank. | *required* |

Returns:

| Type | Description |
| --- | --- |
| `list[Chunk]` | The reranked documents. |

## Usage

```
from datapizza.modules.rerankers.together import TogetherReranker

reranker = TogetherReranker(
    api_key="your-together-api-key",
    model="sentence-transformers/msmarco-bert-base-dot-v5",
    top_n=15,
    threshold=0.3
)

# Rerank documents
query = "How to implement neural networks?"
reranked_results = reranker.rerank(query, document_chunks)
```

## Features

* Access to multiple reranking model options
* Flexible model selection for different use cases
* Score-based filtering with configurable thresholds
* Support for various domain-specific models
* Integration with Together AI's model ecosystem
* Automatic model initialization and management

## Available Models

Common reranking models available through Together AI:

* `sentence-transformers/msmarco-bert-base-dot-v5`
* `sentence-transformers/all-MiniLM-L6-v2`
* `sentence-transformers/all-mpnet-base-v2`
* Custom fine-tuned models for specific domains

## Examples

### Basic Usage

```
import uuid

from datapizza.modules.rerankers.together import TogetherReranker
from datapizza.type import Chunk

# Initialize with specific model
reranker = TogetherReranker(
    api_key="TOGETHER_API_KEY",
    model="Salesforce/Llama-Rank-V1",
    top_n=10,
    threshold=0.4
)

# Sample chunks
chunks = [
    Chunk(id=str(uuid.uuid4()), text="Neural networks are computational models inspired by biological brains..."),
    Chunk(id=str(uuid.uuid4()), text="Deep learning uses multiple layers to learn complex patterns..."),
    Chunk(id=str(uuid.uuid4()), text="Backpropagation is the algorithm used to train neural networks..."),
    Chunk(id=str(uuid.uuid4()), text="The weather is sunny today with mild temperatures..."),
    Chunk(id=str(uuid.uuid4()), text="Convolutional neural networks excel at image recognition tasks...")
]

query = "How do neural networks learn?"

# Rerank based on relevance
reranked_results = reranker.rerank(query, chunks)

# Display results
for i, chunk in enumerate(reranked_results):
    score = chunk.metadata.get('relevance_score', 'N/A')
    print(f"Rank {i+1} (Score: {score}): {chunk.text[:70]}...")
```

---


### Prompt

<a id="39"></a>

## ChatPromptTemplate - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Modules/Prompt/ChatPromptTemplate/

# ChatPromptTemplate

The ChatPromptTemplate class provides utilities for managing prompts, prompt templates, and conversation memory for AI interactions. It helps structure and format prompts for various AI tasks and maintain conversation context.

## datapizza.modules.prompt.ChatPromptTemplate

Bases: `Prompt`

It takes as input a Memory, Chunks, Prompt and creates a Memory
with all existing messages + the user's qry + function\_call\_retrieval +
chunks retrieval.
args:
user\_prompt\_template: str # The user prompt jinja template
retrieval\_prompt\_template: str # The retrieval prompt jinja template

### format

```
format(
    memory=None,
    chunks=None,
    user_prompt="",
    retrieval_query="",
)
```

Creates a new memory object that includes:
- Existing memory messages
- User's query
- Function call retrieval results
- Chunks retrieval results

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `memory` | `Memory | None` | The memory object to add the new messages to. | `None` |
| `chunks` | `list[Chunk] | None` | The chunks to add to the memory. | `None` |
| `user_prompt` | `str` | The user's query. | `''` |
| `retrieval_query` | `str` | The query to search the vectorstore for. | `''` |

Returns:

| Type | Description |
| --- | --- |
| `Memory` | A new memory object with the new messages. |

## Overview

The ChatPromptTemplate module provides utilities for managing prompts and prompt templates in AI pipelines.

```
from datapizza.modules.prompt import prompt
```

**Features:**

* Prompt template management and formatting
* Context-aware prompt construction
* Integration with memory systems for conversation history
* Structured prompt formatting for different AI tasks

## Usage Examples

### Basic Prompt Management

```
import uuid

from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.type import Chunk

# Create structured prompts for different tasks
system_prompt = ChatPromptTemplate(
    user_prompt_template="You are helping with data analysis tasks, this is the user prompt: {{ user_prompt }}",
    retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"
)

print(system_prompt.format(user_prompt="Hello, how are you?", chunks=[Chunk(id=str(uuid.uuid4()), text="This is a chunk"), Chunk(id=str(uuid.uuid4()), text="This is another chunk")]))
```

---


## Tools

<a id="56"></a>

## MCPClient - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Tools/mcp/

# MCPClient

## datapizza.tools.mcp\_client.MCPClient

Helper for interacting with Model Context Protocol servers.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `url` | `str` | The URL of the MCP server. | *required* |
| `command` | `str | None` | The command to run the MCP server. | `None` |
| `headers` | `dict[str, str] | None` | The headers to pass to the MCP server. | `None` |
| `args` | `list[str] | None` | The arguments to pass to the MCP server. | `None` |
| `env` | `dict[str, str] | None` | The environment variables to pass to the MCP server. | `None` |
| `timeout` | `int` | The timeout for the MCP server. | `30` |
| `sampling_callback` | `SamplingFnT | None` | The sampling callback to pass to the MCP server. | `None` |

### a\_list\_prompts `async`

```
a_list_prompts()
```

List the prompts available on the MCP server.

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `A` | `ListPromptsResult` | class:`types.ListPromptsResult` object. |

### a\_list\_tools `async`

```
a_list_tools()
```

List the tools available on the MCP server.

Returns:

| Type | Description |
| --- | --- |
| `list[Tool]` | A list of :class:`Tool` objects. |

### call\_tool `async`

```
call_tool(
    tool_name, arguments=None, progress_callback=None
)
```

Call a tool on the MCP server.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tool_name` | `str` | The name of the tool to call. | *required* |
| `arguments` | `dict[str, Any] | None` | The arguments to pass to the tool. | `None` |
| `progress_callback` | `ProgressFnT | None` | The progress callback to pass to the tool. | `None` |

Returns:

| Type | Description |
| --- | --- |
| `CallToolResult` | The result of the tool call. |

### get\_prompt `async`

```
get_prompt(prompt_name, arguments=None)
```

Get a prompt from the MCP server.

### list\_prompts

```
list_prompts()
```

List the prompts available on the MCP server.

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `A` | `ListPromptsResult` | class:`types.ListPromptsResult` object. |

### list\_resources `async`

```
list_resources()
```

List the resources available on the MCP server.

Returns:

| Name | Type | Description |
| --- | --- | --- |
| `A` | `ListResourcesResult` | class:`types.ListResourcesResult` object. |

### list\_tools

```
list_tools()
```

List the tools available on the MCP server.

Returns:

| Type | Description |
| --- | --- |
| `list[Tool]` | A list of :class:`Tool` objects. |

---

<a id="57"></a>

## DuckDuckGo - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Tools/duckduckgo/

# DuckDuckGo

```
pip install datapizza-ai-tools-duckduckgo
```

## datapizza.tools.duckduckgo.DuckDuckGoSearchTool

Bases: `Tool`

The DuckDuckGo Search tool.
It allows you to search the web for the given query.

### \_\_call\_\_

```
__call__(query)
```

Invoke the tool.

### \_\_init\_\_

```
__init__()
```

Initializes the DuckDuckGoSearch tool.

### search

```
search(query)
```

Search the web for the given query.

## Overview

The DuckDuckGoSearchTool provides web search capabilities using the DuckDuckGo search engine. This tool enables AI models to search for real-time information from the web, making it particularly useful for grounding model responses with current data.

## Features

* **Web Search**: Search the web using DuckDuckGo's search engine
* **Privacy-focused**: Uses DuckDuckGo which doesn't track users
* **Simple Integration**: Easy to integrate with AI agents and pipelines
* **Real-time Results**: Get current web search results

## Usage Example

```
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool

# Initialize the tool
search_tool = DuckDuckGoSearchTool()

# Perform a search
results = search_tool.search("latest AI developments 2024")

# Process results
for result in results:
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"URL: {result.get('href', 'N/A')}")
    print(f"Body: {result.get('body', 'N/A')}")
    print("---")
```

## Integration with Agents

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool

agent = Agent(
    name="agent",
    tools=[DuckDuckGoSearchTool()],
    client=OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4.1"),
)

response = agent.run("What is datapizza? and who are the founders?", tool_choice="required_first")
print(response)
```

---

<a id="55"></a>

## FileSystem - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Tools/filesystem/

# FileSystem

```
pip install datapizza-ai-tools-filesystem
```

## Overview

This tool provides a robust and easy-to-use interface for `datapizza-ai` agents to perform various operations on the local file system. This includes listing, reading, writing, creating, deleting, moving, copying, and precisely replacing content within files.

> **⚠️ Warning: Risk of Data Loss and System Modification**
>
> Operations performed by this tool directly affect your local file system. Using functions like `delete_file`, `delete_directory`, and `write_file` can lead to permanent data loss or unintended system modifications if not used carefully. Exercise extreme caution.

## Features

* **List directories**: `list_directory(path: str)`
* **Read files**: `read_file(file_path: str)`
* **Write files**: `write_file(file_path: str, content: str)`
* **Create directories**: `create_directory(path: str)`
* **Delete files**: `delete_file(file_path: str)`
* **Delete directories**: `delete_directory(path: str, recursive: bool = False)`
* **Move or rename**: `move_item(source_path: str, destination_path: str)`
* **Copy files**: `copy_file(source_path: str, destination_path: str)`
* **Replace with precision**: `replace_in_file(file_path: str, old_string: str, new_string: str)` - Replaces a block of text only if it appears exactly once, requiring context in `old_string` for safety.

## Usage Example

```
import os
import tempfile
import shutil
from datapizza.tools.filesystem import FileSystem

# Initialize the tool
fs_tool = FileSystem()

# Create a temporary directory for the example
temp_dir_path = tempfile.mkdtemp()
print(f"Working in temporary directory: {temp_dir_path}")

# 1. Create a directory
fs_tool.create_directory(os.path.join(temp_dir_path, "my_folder"))

# 2. Write a file
fs_tool.write_file(os.path.join(temp_dir_path, "my_folder", "my_file.txt"), "Hello, world!\nAnother line.")

# 3. Replace content precisely
# The 'old_string' should be unique to avoid errors.
fs_tool.replace_in_file(
    os.path.join(temp_dir_path, "my_folder", "my_file.txt"),
    old_string="Hello, world!",
    new_string="Goodbye, world!"
)

# 4. Read the file to see the change
content = fs_tool.read_file(os.path.join(temp_dir_path, "my_folder", "my_file.txt"))
print(f"File content: {content}")

# Clean up the temporary directory
shutil.rmtree(temp_dir_path)
```

## Integration with Agents

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.filesystem import FileSystem

# 1. Initialize the FileSystem tool
fs_tool = FileSystem()

# 2. Create an agent and provide it with the file system tools
agent = Agent(
    name="filesystem_manager",
    client=OpenAIClient(api_key="YOUR_API_KEY"),
    system_prompt="You are an expert and careful file system manager.",
    tools=[
        fs_tool.list_directory,
        fs_tool.read_file,
        fs_tool.write_file,
        fs_tool.create_directory,
        fs_tool.delete_file,
        fs_tool.delete_directory,
        fs_tool.move_item,
        fs_tool.copy_file,
        fs_tool.replace_in_file,
    ]
)

# 3. Run the agent
# The agent will first read the file, then use 'replace_in_file' with enough context.
response = agent.run("In the file 'test.txt', replace the line 'Hello!' with 'Hello, precisely!'")
print(response)
```

---

<a id="58"></a>

## SQLDatabase - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Tools/SQLDatabase/

# SQLDatabase

```
pip install datapizza-ai-tools-sqldatabase
```

## datapizza.tools.SQLDatabase.SQLDatabase

A collection of tools to interact with a SQL database using SQLAlchemy.
This class is a container for methods that are exposed as tools.

### \_\_init\_\_

```
__init__(db_uri)
```

Initializes the SQLDatabase tool container.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `db_uri` | `str` | The database URI for connection (e.g., "sqlite:///my\_database.db"). | *required* |

### get\_table\_schema

```
get_table_schema(table_name)
```

Returns the schema of a specific table in a human-readable format.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `table_name` | `str` | The name of the table to inspect. | *required* |

### list\_tables

```
list_tables()
```

Returns a newline-separated string of available table names in the database.

### run\_sql\_query

```
run_sql_query(query)
```

Executes a SQL query and returns the result.
For SELECT statements, it returns a JSON string of the rows.
For other statements (INSERT, UPDATE, DELETE), it returns a success message.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `query` | `str` | The SQL query to execute. | *required* |

## Overview

The SQLDatabase tool provides a powerful interface for AI agents to interact with any SQL database supported by SQLAlchemy. This allows models to query structured, relational data to answer questions, providing more accurate and fact-based responses.

## Features

* **Broad Database Support**: Connect to any database with a SQLAlchemy driver (SQLite, PostgreSQL, MySQL, etc.).
* **Schema Inspection**: Allows the agent to view table schemas to understand data structure before querying.
* **Table Listing**: Lets the agent list all available tables to get context of the database.

## Integration with Agents

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.SQLDatabase import SQLDatabase

db_uri = "sqlite:///company.db"

# 1. Initialize the SQLDatabase tool
db_tool = SQLDatabase(db_uri=db_uri)

# 2. Create an agent and provide it with the database tool's methods
agent = Agent(
    name="database_expert",
    client=OpenAIClient(api_key="YOUR_API_KEY"),
    system_prompt="You are a database expert. Use the available tools to answer questions about the database.",
    tools=[
        db_tool.list_tables,
        db_tool.get_table_schema,
        db_tool.run_sql_query
    ]
)

# 3. Run the agent
response = agent.run("How many people work in the Engineering department?")
print(response)
```

---

<a id="54"></a>

## WebFetch - Datapizza AI

> 🔗 https://docs.datapizza.ai/0.0.9/API%20Reference/Tools/web_fetch/

# WebFetch

```
pip install datapizza-ai-tools-web-fetch
```

## datapizza.tools.web\_fetch.base.WebFetchTool

Bases: `Tool`

The Web Fetch tool.
It allows you to fetch the content of a given URL with configurable timeouts
and specific error handling.

### \_\_call\_\_

```
__call__(url)
```

Invoke the tool.

### \_\_init\_\_

```
__init__(timeout=None, user_agent=None)
```

Initializes the WebFetchTool.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `timeout` | `float | None` | The timeout for the request in seconds. | `None` |
| `user_agent` | `str | None` | The User-Agent header to use for the request. | `None` |

## Overview

The WebFetch tool provides a simple and robust way for AI agents to retrieve content from a given URL. It allows models to access live information from the internet, which is crucial for tasks requiring up-to-date data.

## Features

* **Live Web Access**: Fetches content from any public URL.
* **Error Handling**: Gracefully handles common HTTP errors (e.g., timeouts, 404, 503) and reports them clearly.
* **Configurable**: Allows setting custom timeouts and User-Agent strings.
* **Simple Integration**: As a callable tool, it integrates seamlessly with `datapizza-ai` agents.

## Usage Example

```
from datapizza.tools.web_fetch import WebFetchTool

# Initialize the tool
fetch_tool = WebFetchTool()

# Fetch content from a URL by calling the tool instance
content = fetch_tool("https://example.com")

print(content)
```

## Integration with Agents

```
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.web_fetch import WebFetchTool

# 1. Initialize the WebFetchTool, optionally with a custom timeout
web_tool = WebFetchTool(timeout=15.0)

# 2. Create an agent and provide it with the tool
agent = Agent(
    name="web_researcher",
    client=OpenAIClient(api_key="YOUR_API_KEY"),
    system_prompt="You are a research assistant. Use the web_fetch tool to get information from URLs to answer questions.",
    tools=[web_tool]
)

# 3. Run the agent to summarize a web page
response = agent.run("Please summarize the content of https://loremipsum.io/")
print(response.text)
```

---
