# Streaming

Streaming allows you to receive responses in real-time as they're generated, providing a better user experience for long responses and interactive applications.

## Why Use Streaming?

- **Real-time feedback**: Users see responses as they're generated
- **Better UX**: Reduces perceived latency for long responses
- **Progressive display**: Show partial results immediately
- **Interruptible**: Can stop generation early if needed

## Basic Streaming

### Synchronous Streaming

```python
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

```python
import asyncio

async def async_stream_example():
    async for chunk in client.a_stream_invoke("Explain quantum computing in simple terms"):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
    print()  # New line when complete

# Run the async function
asyncio.run(async_stream_example())
```
