# yeni eklenen modullere göre tüm modüller import edilecek !!!

import glob
from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pymupdf
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# PDF dosyasını yükleme
pdf_folders_path =glob.glob("../../knowledge_base/*")
text_loader_kwargs = {'encoding': 'utf-8'}
print(pdf_folders_path)


chunked_documents = []

for pdf_folder in pdf_folders_path:
    
    print(f"Processing folder: {pdf_folder}")

    loader = DirectoryLoader(pdf_folder, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    folder_docs = loader.load()

    
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))


    for pdf_path in pdf_files:

        #PDF filename
        pdf_filename = os.path.basename(pdf_path)
        
        # PDF dosyasını aç
        pdf_open = pymupdf.open(pdf_path)
        
        # Table of Contents (TOC) al
        toc = pdf_open.get_toc()

        # PyMuPDF metadata'yı al
        pdf_metadata = pdf_open.metadata

        for i, item in enumerate(toc):
            heading = item[1]
            start_page = item[2]

            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = pdf_open.page_count  # Son başlıksa PDF'in son sayfasına kadar
    
            # Sonraki başlığın sayfasına kadar olan kısmı almak
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = pdf_open.page_count -  1   # Son başlıksa PDF'in son sayfasına kadar

            # Metni birleştirerek chunk oluştur
            chunk_text = ""
            for page_num in range(start_page, end_page):  # PyMuPDF için 0-index
                chunk_text += pdf_open[page_num].get_text()

            dynamic_metadata = {key: value for key, value in pdf_metadata.items() if value}
            
        # İlk chunk'ları oluştur ve listeye ekle
            chunked_documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={"source": pdf_filename ,"heading": heading, "start_page": start_page, "end_page": end_page,**dynamic_metadata}
                )
            )
            
# CharacterTextSplitter kullanarak ek parçalama
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# TOC'ye göre bölünmüş metinleri tekrar split ediyoruz
final_chunks = text_splitter.split_documents(chunked_documents)


# Create our Chroma vectorstore!
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


db_name = "../vector-db"



if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create our Chroma vectorstore!

vectorstore = Chroma.from_documents(documents=final_chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")