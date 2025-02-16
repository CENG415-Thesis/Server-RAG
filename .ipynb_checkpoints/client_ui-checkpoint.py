import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama 

# Mevcut vektör veritabanının adı
DB_NAME = "vector-db"

# Vektör veritabanını yükleme
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Ollama'yı doğrudan Python içinde kullan
llm = Ollama(model="llama3.2")

def query_rag_pipeline(user_query):
    retrieved_docs = vectorstore.similarity_search(user_query, k=10)
    combined_context = " ".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are a telecom assistant. Use the provided context to answer the question.

    Context: {combined_context}

    Question: {user_query}

    Answer:
    """

    response = llm.invoke(prompt)  # Doğrudan LLM çağrısı
    return response

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
