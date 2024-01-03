from .embeddings import getEmbedModel
from llama_index.vector_stores import ChromaVectorStore # TODO: Replace with the abstract type
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms.utils import LLMType, resolve_llm
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

"""LLM Augmented with additional documents. Defaults to using Llama-2-7b-chat but can be extended to alternative LLMs"""
class AugmentedLLM:
    def __init__(self, vector_store: ChromaVectorStore, hf_token: str=None, hf_embedding_model: str=None, llm: LLMType = None):
        self.embedding_model = getEmbedModel(hf_embedding_model)
        self.hf_token = hf_token
        self.llm = self.__getLLM(llm)
        self.index = self.__createVectorDatabase()
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
        # Default to llama-2 7B. Swap this out for your preferred model.
        self.model = "meta-llama/Llama-2-7b-chat-hf"

        # Must have hf token specified as this model requires access 
        if self.hf_token is None:
            print('Please specify huggingface token in order to use default model')
            exit()

        SYSTEM_PROMPT = """You are an AI assistant that answers questions. Use the provided context to answer if possible, otherwise defer to other knowledge if the provided context is not helpful
        """

        query_wrapper_prompt = PromptTemplate(
            "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
        )

        llm = HuggingFaceLLM(
            context_window=4096,
            
            generate_kwargs={"temperature": 0.3},
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=self.model,
            model_name=self.model,
            model_kwargs={"token": self.hf_token},
            tokenizer_kwargs={"token": self.hf_token},
            # device_map="auto",
            # change these settings below depending on your GPU
            # model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
            # model_kwargs={},
        )

        return llm

    # TODO: change vector store type
    def __createVectorDatabase(self, vectorStore: ChromaVectorStore):        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create service context
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embedding_model,
        )
            
        # load your index from stored vectors
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, service_context=service_context
        )

        return index


    def query(self, prompt: str) -> str:
        return self.query_engine.query(prompt)