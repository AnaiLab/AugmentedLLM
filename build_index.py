import chromadb
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from config import Config
from utils.embeddings import getEmbedModel

# Edit these if you don't want to use the config file
documents_dir = Config.DOCUMENTS_DIR
db_save_dir = Config.CHROMA_DIR
collection_name = Config.CHROMA_COLLECTION
embedding_model = Config.EMBEDDING_MODEL

# load some documents
documents = SimpleDirectoryReader(documents_dir).load_data()

# initialize client, setting path to save data
db = chromadb.PersistentClient(path=db_save_dir)

# create collection
chroma_collection = db.get_or_create_collection(collection_name)

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(llm=None, embed_model=getEmbedModel(embedding_model))
index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
