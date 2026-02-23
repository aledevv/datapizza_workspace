# ImageRAGPrompt

Specialized prompting utilities for Retrieval-Augmented Generation (RAG) with image content. The ImageRAGPrompt class provides multimodal content integration for vision-language models.

<!-- prettier-ignore -->
::: datapizza.modules.prompt.ImageRAGPrompt
    options:
        show_source: false


## Overview

```python
from datapizza.modules.prompt.image_rag import ImageRAGPrompt

# Initialize image RAG prompt handler
image_rag = ImageRAGPrompt()
```

**Features:**

- Image-aware RAG prompt construction
- Multimodal content integration
- Context preservation for image-text interactions
- Optimized prompting for vision-language models

## Usage Examples

### Basic Image RAG Usage
```python
from datapizza.modules.prompt.image_rag import ImageRAGPrompt
from datapizza.type import Media

# Initialize image RAG prompt
image_rag = ImageRAGPrompt()

# Create multimodal RAG prompt
media_content = Media(data=image_data, media_type="image/png")
rag_prompt = image_rag.create_rag_prompt(
    query="What does this chart show?",
    retrieved_context=text_context,
    images=[media_content],
    instructions="Analyze both the text context and image content"
)
```
