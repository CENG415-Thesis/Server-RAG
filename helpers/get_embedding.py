# not used
from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    """
    Initializes and returns an embedding model.

    The current implementation uses the OllamaEmbeddings model with the 
    "nomic-embed-text" configuration. This can be modified to use other 
    embedding models, such as HuggingFace Embeddings.

    Returns:
        OllamaEmbeddings: An instance of the OllamaEmbeddings model.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    #Try another embeddings, HuggingFace Embeddings                                
    
    return embeddings