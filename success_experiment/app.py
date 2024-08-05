from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import sys
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core import StorageContext ,load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage",)


llm = Ollama(model="llama3", request_timeout=120.0)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# load index
index = load_index_from_storage(storage_context,embed_model=embed_model)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

vector_retriever = index.as_retriever(similarity_top_k=1)
# query engine without RecursiveRetriever
vector_query_engine = index.as_query_engine(similarity_top_k=1,llm=llm)

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},

    verbose=True
)
query_engine = RetrieverQueryEngine.from_args(recursive_retriever,llm=llm)

response = query_engine.query("Summarize Tamarac")
print(response)