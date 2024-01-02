from langchain.embeddings import HuggingFaceEmbeddings

def getEmbedModel(self, hf_embedding_model: str):
    # Local defualt
    if(hf_embedding_model is None or hf_embedding_model.lower() == 'local'):
        return 'local'

    # Alternatively, if a model is specified, retrieve it
    return HuggingFaceEmbeddings(model_name=hf_embedding_model)