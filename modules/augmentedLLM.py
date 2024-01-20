from .embeddings import getEmbedModel
from llama_index.vector_stores import ChromaVectorStore # TODO: Replace with the abstract type
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms.utils import LLMType, resolve_llm
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
import torch

"""LLM Augmented with additional documents. Defaults to using Llama-2-7b-chat but can be extended to alternative LLMs"""
class AugmentedLLM:
    def __init__(self, vector_store: ChromaVectorStore, hf_token: str=None, hf_embedding_model: str=None, llm: LLMType = None):
        self.embedding_model = getEmbedModel(hf_embedding_model)
        self.hf_token = hf_token
        self.llm = self.__getLLM(llm)
        self.index = self.__createVectorDatabase(vector_store)
        self.query_engine = self.index.as_query_engine()

    def __getEmbedModel(self, hf_embedding_model: str):
        # Local defualt
        if(hf_embedding_model is None or hf_embedding_model.lower() == 'local'):
            return 'local'

        # Alternatively, if a model is specified, retrieve it
        return HuggingFaceEmbeddings(model_name=hf_embedding_model)
    
    def __getLLM(self, llm: LLMType):
        # If model is unspecified, use default model
        if llm is None:
            return self.__getDefaultLLM()

        else:
            return llm
    

    def __getDefaultLLM(self):
        model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true"

        llm = LlamaCPP(
            model_url=model_url,

            # Note, you can use pre-downloaded weights rather than downloading from HF by passing in
            model_path=None,
            temperature=0.2,
            max_new_tokens=256,
            context_window=4000,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": 4}, 

            # Match expected Llama prompt format
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )

        return llm

    # TODO: change vector store type
    def __createVectorDatabase(self, vectorStore: ChromaVectorStore):        
        storage_context = StorageContext.from_defaults(vector_store=vectorStore)

        # Create service context
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding_model,
        )
            
        # load your index from stored vectors
        index = VectorStoreIndex.from_vector_store(
            vectorStore, storage_context=storage_context, service_context=service_context
        )

        return index


    def query(self, prompt: str) -> str:
        return self.query_engine.query(prompt)