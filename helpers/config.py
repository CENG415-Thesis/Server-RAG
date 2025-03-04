from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from helpers.constants import DB_NAME
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration objects
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = Ollama(model="llama3.2")
memory = ConversationBufferMemory()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)