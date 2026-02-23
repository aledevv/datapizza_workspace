# DuckDuckGo

```bash
pip install datapizza-ai-tools-duckduckgo
```

<!-- prettier-ignore -->
::: datapizza.tools.duckduckgo.DuckDuckGoSearchTool
    options:
        show_source: false

## Overview

The DuckDuckGoSearchTool provides web search capabilities using the DuckDuckGo search engine. This tool enables AI models to search for real-time information from the web, making it particularly useful for grounding model responses with current data.

## Features

- **Web Search**: Search the web using DuckDuckGo's search engine
- **Privacy-focused**: Uses DuckDuckGo which doesn't track users
- **Simple Integration**: Easy to integrate with AI agents and pipelines
- **Real-time Results**: Get current web search results

## Usage Example

```python
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

```python
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
