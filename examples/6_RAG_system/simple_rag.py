import os

from datapizza.clients.google import GoogleClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders.cohere import CohereEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from dotenv import load_dotenv

load_dotenv()

embedder = CohereEmbedder(
    api_key=os.getenv("COHERE_API_KEY"),
    base_url=os.getenv("COHERE_ENDPOINT"),
    model_name="embed-v4.0",
    input_type="query",
)

retriever = QdrantVectorstore(
    host="localhost",
    port=6333,
)

retriever.create_collection(
    collection_name="knowledge_base",
    vector_config=[VectorConfig(dimensions=1536, name="vector")],
)

prompt_template = ChatPromptTemplate(
    user_prompt_template="User question: {{user_prompt}}\n: ",
    retrieval_prompt_template="Retrieved content: \n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}",
)

client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash",
)

dag_pipeline = DagPipeline()
dag_pipeline.add_module("embedder", embedder)
dag_pipeline.add_module("retriever", retriever)
dag_pipeline.add_module("prompt", prompt_template)
dag_pipeline.add_module("generator", client)

dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
dag_pipeline.connect("retriever", "prompt", target_key="chunks")
dag_pipeline.connect("prompt", "generator", target_key="memory")

query = "List the rare langauges"

result = dag_pipeline.run(
    {
        "embedder": {"text": query},
        "prompt": {"user_prompt": query},
        "retriever": {"collection_name": "video_tutorial_ingestion", "k": 5},
        "generator": {"input": query},
    }
)

print(f"Generated response: {result['generator']}")