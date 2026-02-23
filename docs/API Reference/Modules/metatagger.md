# Metatagger

Metataggers are pipeline components that add metadata tags to content chunks using language models. They analyze text content and generate relevant keywords, tags, or other metadata to enhance content discoverability and organization.

<!-- prettier-ignore -->
::: datapizza.modules.metatagger.KeywordMetatagger
    options:
        show_source: false



A metatagger that uses language models to generate keywords and metadata for text chunks.

```python
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

- Processes chunks in parallel for better performance
- Configurable prompts for different keyword extraction strategies
- Adds generated keywords to chunk metadata
- Supports custom metadata field naming
- Handles both individual chunks and lists of chunks
- Uses memory-based conversation for consistent prompting

**Input/Output:**

- Input: `Chunk` objects or lists of `Chunk` objects
- Output: Same `Chunk` objects with additional metadata containing generated keywords

## Usage Examples

### Basic Keyword Extraction
```python
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
