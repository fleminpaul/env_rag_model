from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import sys
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext ,load_index_from_storage
from flask import Flask, request,jsonify
import json

from llama_index.core.memory import ChatMemoryBuffer

app = Flask(__name__)

def initialze():
    llm = Ollama(model="llama3", request_timeout=120.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context,embed_model=embed_model)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    vector_retriever = index.as_chat_engine(llm=llm, similarity_top_k=1)
    # query engine without RecursiveRetriever
    vector_query_engine = index.as_chat_engine(chat_mode="condense_plus_context", similarity_top_k=1,llm=llm)

    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        verbose=True
    )
    query_engine = RetrieverQueryEngine.from_args(recursive_retriever,llm=llm)
    memory = ChatMemoryBuffer.from_defaults(token_limit=8000)

    chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are Product Support Assistant, who is polite in answering any questions related to Envestnet, Tamarac"
        " You cannot mention anything to user related to you limitations . If user ask to create a support ticket. Ask for all required details to create a support ticket"
        " Once all the information are recieved create a JSON of format {\"Description\":\"<heading of the issue>\"},{\"IssueDetails\":\"<Details>\"} append the json string ###SUPPORT_TICKET###"
        " If you don't know answer reply honstely and donot hallucinate"
    ),llm=llm
    )
    print("Initialized")
    return chat_engine

query_engine=initialze()

@app.route("/query", methods=["POST"])
def query():
    print("Post /query called")
    print(request.json)
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = query_engine.chat(query)

    print(response)

    response_answer = {"answer": str(response)}
    return jsonify(response_answer)

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
