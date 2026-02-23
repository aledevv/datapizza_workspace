# DagPipeline

The `DagPipeline` class allows you to define and execute a series of processing steps (modules) organized as a Directed Acyclic Graph (DAG). Modules typically inherit from `datapizza.core.models.PipelineComponent` or are simple callables. This enables complex workflows where the output of one module can be selectively used as input for others.

## Core Concepts

### Modules

Modules are the building blocks of the pipeline. They are typically instances of classes inheriting from `datapizza.core.models.PipelineComponent` (which requires implementing a `run` and  `a_run` method), `datapizza.core.models.ChainableProducer` (which exposes an `as_module_component` method returning a `PipelineComponent`), or simply Python callables.

```python
from datapizza.core.models import PipelineComponent
from datapizza.pipeline import DagPipeline

class MyProcessingStep(PipelineComponent):
    # Inheriting from PipelineComponent provides the __call__ wrapper for logging
    def _run(self, input_data: str) -> str:
        return something

    async _a_run(self, something: str) -> str:
        return await do_stuff()

```

### Connections

Connections define the flow of data between modules. You specify which module's output connects to which module's input.

-   **`from_node_name`**: The name of the source module.
-   **`to_node_name`**: The name of the target module.
-   **`source_key`** (Optional): If the source module's `process` method (or callable) returns a dictionary, this key specifies which value from the dictionary should be passed. If `None`, the entire output of the source module is passed.
-   **`target_key`** : This key specifies the argument name in the target module's `process` method (or callable) that should receive the data. If `None`, and the source output is *not* a dictionary, the data is passed as the first non-`self` argument to the target's `_run` method/callable. If `None` and the source output *is* a dictionary, its key-value pairs are merged into the target's input keyword arguments.

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.core.models import PipelineComponent
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

client = OpenAIClient(api_key="OPENAI_API_KEY", model="gpt-4o-mini")
vector_store = QdrantVectorstore(location=":memory:")
vector_store.create_collection(collection_name="my_documents", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])

pipeline = DagPipeline()

pipeline.add_module("rewriter", ToolRewriter(client=client, system_prompt="rewrite the query to perform a better search in a vector database"))
pipeline.add_module("embedder", OpenAIEmbedder(api_key="OPENAI_API_KEY", model_name="text-embedding-3-small"))
pipeline.add_module("vector_store", vector_store)
pipeline.add_module("prompt_template", ChatPromptTemplate(user_prompt_template = "this is a user prompt: {{ user_prompt }}", retrieval_prompt_template = "{% for chunk in chunks %} Relevant chunk: {{ chunk.text }} \n\n {% endfor %}"))
pipeline.add_module("llm", OpenAIClient(model = "gpt-4o-mini", api_key = "OPENAI_API_KEY"))


pipeline.connect("rewriter", "embedder", target_key="text")
pipeline.connect("embedder", "vector_store", target_key="query_vector")
pipeline.connect("vector_store", "prompt_template", target_key="chunks")
pipeline.connect("prompt_template", "llm", target_key="memory")
```

## Running the Pipeline

The `run` method executes the pipeline based on the defined connections. It requires an initial `data` dictionary which provides the missing input arguments for the nodes that require them.

The keys of this dictionary should match the names of the modules requiring initial input, and the values should be dictionaries mapping argument names to values for their respective `process` methods (or callables).

```python
user_input = "tell me something about this document"
res = pipeline.run(
    {
        "rewriter": {"user_prompt": user_input},

        # Embedder doesn't require any input because it's provided by the rewriter

        "prompt_template": {"user_prompt": user_input},  # Prompt template requires user_prompt
        "vector_store": {
            "collection_name": "my_documents",
            "k": 10,
        },
        "llm": {
            "input": user_input,
            "system_prompt": "You are a helpful assistant. try to answer user questions given the context",
        },
    }
)
result = res.get("llm").text
print(result)
```

The pipeline automatically determines the execution order based on dependencies. It executes modules by calling their `run` method only when all their prerequisites (connected `from_node_name` modules) have completed successfully.



### Async run

Pipeline support async run with  `a_run`
With async run, the pipeline will call a_run of modules.

This only works if you are using a remote qdrant server. The in-memory qdrant function does not work with asynchronous execution.
```python

res = await pipeline.a_run(
    {
        "rewriter": {"user_prompt": user_input},
        "prompt_template": {"user_prompt": user_input},
        "vector_store": {
            "collection_name": "datapizza",
            "k": 10,
        },
        "llm": {
            "input": user_input,
            "system_prompt": "You are a helpful assistant. try to answer user questions given the context",
        },
    }
)

```


## Configuration via YAML

Pipelines can be defined entirely using a YAML configuration file, which is loaded using the `from_yaml` method. This is useful for separating pipeline structure from code.

The YAML structure includes sections for `clients` (like LLM providers), `modules`, and `connections`.

```python
from datapizza.pipeline import DagPipeline

pipeline = DagPipeline().from_yaml("dag_pipeline.yaml")
user_input = "tell me something about this document"
res = pipeline.run(
    {
        "rewriter": {"user_prompt": user_input},
        "prompt_template": {"user_prompt": user_input},
        "vector_store": {"collection_name": "my_documents","k": 10,},
        "llm": {"input": user_input,"system_prompt": "You are a helpful assistant. try to answer user questions given the context",},
    }
)
result = res.get("llm").text
print(result)
```

### Example YAML (`dag_config.yaml`)

```yaml
dag_pipeline:
  clients:
    openai_client:
      provider: openai
      model: "gpt-4o-mini"
      api_key: ${OPENAI_API_KEY}
    google_client:
      provider: google
      model: "gemini-2.0"
      api_key: ${GOOGLE_API_KEY}
    openai_embedder:
      provider: openai
      model: "text-embedding-3-small"
      api_key: ${OPENAI_API_KEY}

  modules:
    - name: rewriter
      type: ToolRewriter
      module: datapizza.modules.rewriters
      params:
        client: openai_client
        system_prompt: "rewrite the query to perform a better search in a vector database"
    - name: embedder
      type: ClientEmbedder
      module: datapizza.embedders
      params:
        client: openai_embedder
    - name: vector_store
      type: QdrantVectorstore
      module: datapizza.vectorstores.qdrant
      params:
        host: localhost
    - name: prompt_template
      type: ChatPromptTemplate
      module: datapizza.modules.prompt
      params:
        user_prompt_template: "this is a user prompt: {{ user_prompt }}"
        retrieval_prompt_template: "{% for chunk in chunks %} Relevant chunk: {{ chunk.text }} \n\n {% endfor %}"
    - name: llm
      type: OpenAIClient
      module: datapizza.clients.openai
      params:
        model: "gpt-4o-mini"
        api_key: ${OPENAI_API_KEY}

  connections:

    - from: rewriter
      to: embedder
      target_key: text
    - from: embedder
      to: vector_store
      target_key: query_vector
    - from: vector_store
      to: prompt_template
      target_key: chunks
    - from: prompt_template
      to: llm
      target_key: memory
```

**Key points for YAML configuration:**

-   **Environment Variables**: Use `${VAR_NAME}` syntax to load sensitive information like API keys from environment variables.
-   **Clients**: Define clients once and reference them by name in module `params`.
-   **Module Loading**: Specify the `module` path and `type` (class name) for dynamic loading. The class should generally be a `PipelineComponent`.
-   **Parameters**: `params` are passed directly to the module's constructor.
-   **Connections**: Define data flow similarly to the programmatic `connect` method.
