import os

# Replace with path to your RTF documents and desired output location (Step 2.1.1)
rtf_file_dir = './articles_rtf/'
converted_file_dir = './articles/'

# Replace with the path to the folder containing your documents (Step 2.2.1)
article_dir = './articles/'

embedding_model = 'local'

# Replace with hf API key you would like to use
huggingface_key = 'your-key-here'


# Replace with openAI key if you would like to test against OpenAI models
openai_key = ''

# Model output location
output_dir = './output/'


# Do not edit the code below
class Config:
    RTF_FILE_DIR = rtf_file_dir
    CONVERTED_FILE_DIR = converted_file_dir
    DOCUMENTS_DIR = article_dir
    CHROMA_DIR = './vector_db'
    CHROMA_COLLECTION = 'SurgicalManuscripts'
    OPENAI_KEY = openai_key
    HUGGINGFACE_KEY = huggingface_key
    EMBEDDING_MODEL = embedding_model











