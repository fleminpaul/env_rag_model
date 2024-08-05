from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex,StorageContext
# from llama_index.core.retrievers import RecursiveRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core import SimpleDirectoryReader
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import logging
import sys

# import nltk

# # Download the required resource if not already done
# nltk.download('averaged_perceptron_tagger')

#data path
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # initialize client, setting path to save data
# db = chromadb.PersistentClient(path="./chroma_db")
# # create collection
# chroma_collection = db.get_or_create_collection("product_documentation")

# assign chroma as the vector_store to the context
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
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
# vector_retriever = vector_index.as_retriever(similarity_top_k=1)
# # query engine without RecursiveRetriever
# vector_query_engine = vector_index.as_query_engine(similarity_top_k=1,llm=llm)

# recursive_retriever = RecursiveRetriever(
#     "vector",
#     retriever_dict={"vector": vector_retriever},
#     node_dict=node_mappings,
#     verbose=True
# )
# query_engine = RetrieverQueryEngine.from_args(recursive_retriever,llm=llm)
storage_context.persist()

# response = query_engine.query("what is PE Ratio?")
# print(response)