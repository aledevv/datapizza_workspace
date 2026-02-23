# Rewriters

Rewriters are pipeline components that transform and rewrite text content using language models. They can modify content style, format, tone, or structure while preserving meaning and important information.

<!-- prettier-ignore -->
::: datapizza.modules.rewriters.ToolRewriter
    options:
        show_source: false




A rewriter that uses language models to transform text content with specific instructions and tools.

```python
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

- Flexible content transformation with custom instructions
- Support for various rewriting tasks (summarization, style changes, format conversion)
- Integration with tool calling for enhanced capabilities
- Preserves important information while transforming presentation
- Supports both sync and async processing
- Configurable prompting for different rewriting strategies
