import streamlit as st
from langchain_chroma import Chroma
#from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# Mevcut vektör veritabanının adı
DB_NAME = "vector-db"

# Vektör veritabanını yükleme
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# LLM modelini yükleme
llm = ChatOllama(model="llama3.2", base_url="http://45.145.22.22:11434", temperature=0, system="You are a telecom assistant.", num_gpu_layers=0, backend="cpu")

def query_rag_pipeline(user_query):
    retrieved_docs = vectorstore.similarity_search(user_query, k=10)
    combined_context = " ".join([doc.page_content for doc in retrieved_docs])
    
    input_message = f"Context: {combined_context}\n\nQuestion: {user_query}"
    response = llm.invoke(input_message)
    return response.content

# Streamlit Arayüzü
st.title("Client-Side RAG Assistant")
st.markdown("Bu asistan, mevcut vektör veritabanını kullanarak sorgularınıza yanıt verir.")

user_query = st.text_input("Sorgunuzu girin:")

if st.button("Yanıtla"):
    if user_query:
        with st.spinner("Yanıt oluşturuluyor..."):
            response = query_rag_pipeline(user_query)
            st.subheader("Yanıt:")
            st.write(response)
    else:
        st.warning("Lütfen bir soru girin.")
