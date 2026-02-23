# Treebuilder

Treebuilders are pipeline components that construct hierarchical tree structures (Node objects) from various types of content. They convert flat or unstructured content into organized, nested representations that facilitate better processing and understanding.

<!-- prettier-ignore -->
::: datapizza.modules.treebuilder.LLMTreeBuilder
    options:
        show_source: false



A treebuilder that uses language models to analyze content and create hierarchical structures based on semantic understanding.

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.modules.treebuilder import LLMTreeBuilder

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o")
treebuilder = LLMTreeBuilder(client=client)

flat_content = "This is a flat piece of content. It should be converted into a hierarchical structure."

structured_document = treebuilder.parse(flat_content)

print(structured_document)
```

**Features:**

- Semantic understanding of content organization
- Configurable tree depth and structure rules
- Support for various content types (articles, reports, manuals, etc.)
- Preserves original content while adding hierarchical organization
- Metadata extraction and tagging during structure creation
- Supports both sync and async processing

## Usage Examples

### Basic Tree Structure Creation
```python
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
