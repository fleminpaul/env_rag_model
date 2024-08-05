from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import logging
import sys

#data path
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

storage_context = StorageContext.from_defaults()

data_path="C:\\Users\\Flemin\\OneDrive\\Desktop\\Data\\"

reader = SimpleDirectoryReader(input_dir=data_path,recursive=True,required_exts=[".html",".htm"])
documents = reader.load_data(show_progress=True)

llm = Ollama(model="llama3", request_timeout=120.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

parser = UnstructuredElementNodeParser(llm=llm)
nodes = parser.get_nodes_from_documents(documents)
base_nodes, node_mappings = parser.get_base_nodes_and_mappings(nodes)

# construct top-level vector index + query engine
vector_index = VectorStoreIndex(nodes,embed_model=embed_model,storage_context=storage_context)

storage_context.persist()

