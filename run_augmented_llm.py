from modules.augmentedLLM import AugmentedLLM

import chromadb
from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from config import Config

# Edit these if you don't want to use the config file
documents_dir = Config.DOCUMENTS_DIR
db_save_dir = Config.CHROMA_DIR
collection_name = Config.CHROMA_COLLECTION

# Retrieve previously made chroma vector db
db = chromadb.PersistentClient(path=db_save_dir)
chroma_collection = db.get_or_create_collection(collection_name)

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Note, alternatively you can define an LLM object and pass in to use instead of the default
augmentedLLM = AugmentedLLM(vector_store, llm=None)

while True:
    query = input('\n Query: \n')
    print(augmentedLLM.query(query))
