# ChatPromptTemplate

The ChatPromptTemplate class provides utilities for managing prompts, prompt templates, and conversation memory for AI interactions. It helps structure and format prompts for various AI tasks and maintain conversation context.

<!-- prettier-ignore -->
::: datapizza.modules.prompt.ChatPromptTemplate
    options:
        show_source: false

## Overview

The ChatPromptTemplate module provides utilities for managing prompts and prompt templates in AI pipelines.

```python
from datapizza.modules.prompt import prompt
```

**Features:**

- Prompt template management and formatting
- Context-aware prompt construction
- Integration with memory systems for conversation history
- Structured prompt formatting for different AI tasks

## Usage Examples

### Basic Prompt Management
```python
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
