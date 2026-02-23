# TogetherReranker

A reranker that uses Together AI's API for document reranking with various model options.

## Installation

```bash
pip install datapizza-ai-rerankers-together
```
<!-- prettier-ignore -->
::: datapizza.modules.rerankers.together.TogetherReranker
    options:
        show_source: false



## Usage

```python
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

- Access to multiple reranking model options
- Flexible model selection for different use cases
- Score-based filtering with configurable thresholds
- Support for various domain-specific models
- Integration with Together AI's model ecosystem
- Automatic model initialization and management

## Available Models

Common reranking models available through Together AI:

- `sentence-transformers/msmarco-bert-base-dot-v5`
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`
- Custom fine-tuned models for specific domains

## Examples

### Basic Usage

```python
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
