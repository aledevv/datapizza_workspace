# Tools

Tools allow AI models to call external functions, enabling them to perform actions, retrieve data, and interact with external systems.

## Basic Tool Usage

### Simple Tool

```python
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

```python
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

```python
response = client.invoke(
    "Hello, how are you?",
    tools=[get_weather],
    tool_choice="auto"  # Model may or may not use tools
)
```

### Required
Force the model to use a tool:

```python
response = client.invoke(
    "Get weather information",
    tools=[get_weather],
    tool_choice="required"  # Model must use a tool
)
```

### None
Disable tool usage:

```python
response = client.invoke(
    "What's the weather like?",
    tools=[get_weather],
    tool_choice="none"  # Model won't use tools
)
```

### Specific Tool
Force a specific tool:


```python
response = client.invoke(
    "Check the weather",
    tools=[get_weather, get_time],
    tool_choice=["get_weather"]  # Only use this specific tool
)
```
