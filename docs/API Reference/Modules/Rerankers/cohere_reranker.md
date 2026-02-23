# CohereReranker

A reranker that uses Cohere's reranking API to score and reorder documents based on query relevance.

## Installation

```bash
pip install datapizza-ai-rerankers-cohere
```

<!-- prettier-ignore -->
::: datapizza.modules.rerankers.cohere.CohereReranker
    options:
        show_source: false


## Usage

```python
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

- High-quality semantic reranking using Cohere's models
- Configurable result count and score thresholds
- Support for both sync and async processing
- Automatic relevance scoring for retrieved content
- Integration with Cohere's latest reranking models
- Flexible endpoint configuration for different Cohere services

## Examples

### Basic Usage

```python
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
