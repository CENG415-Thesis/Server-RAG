import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_ollama import ChatOllama
from get_upload_func import save_uploaded_file

# Mevcut vektör veritabanının adı
DB_NAME = "vector-db"

# Embed yöntemleri
embed_methods = {
    "FastEmbed": FastEmbedEmbeddings(device="cpu")
}

st.title("Server-Side RAG Assistant")
st.markdown("Bu sistem, yeni PDF'leri yükleyip embedding işlemi yaparak mevcut veritabanına eklemenizi sağlar.")

# Kullanıcıdan embed yöntemi seçmesini iste (Sorgunun üstüne konumlandırıldı)
selected_embed = st.selectbox("Embedding Yöntemi Seçin", list(embed_methods.keys()))
embeddings = embed_methods[selected_embed]

# Vektör veritabanını yükleme
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# LLM modelini yükleme
llm = ChatOllama(model="llama3.2", base_url="http://45.145.22.22:11434", temperature=0, system="You are a telecom assistant.", num_gpu_layers=0)

# PDF Yükleme Alanı
uploaded_file = st.file_uploader("Yeni PDF Yükleyin", type=["pdf"])

if uploaded_file:
    save_uploaded_file(uploaded_file, embeddings, DB_NAME)
    st.success(f"{uploaded_file.name} başarıyla vektör veritabanına eklendi!")

# Kullanıcıdan sorgu alma
user_query = st.text_input("Sorgunuzu girin:")

def query_rag_pipeline(user_query):
    retrieved_docs = vectorstore.similarity_search(user_query, k=10)
    combined_context = " ".join([doc.page_content for doc in retrieved_docs])
    
    input_message = f"Context: {combined_context}\n\nQuestion: {user_query}"
    response = llm.invoke(input_message)
    return response.content

if st.button("Yanıtla"):
    if user_query:
        with st.spinner("Yanıt oluşturuluyor..."):
            response = query_rag_pipeline(user_query)
            st.subheader("Yanıt:")
            st.write(response)
    else:
        st.warning("Lütfen bir soru girin.")
