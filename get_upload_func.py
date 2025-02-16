import os
import pymupdf
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def save_uploaded_file(pdf_file, embed_method, db_name="vector-db", uploaded_files_dir="uploaded_files"):
    """
    Yüklenen PDF dosyasını belirtilen klasöre kaydeder, metni işler ve vektör veritabanına ekler.
    
    :param pdf_file: Gradio tarafından döndürülen NamedString nesnesi (pdf_file.name içinden yol alınır)
    :param embed_method: Seçilen embedding yöntemi
    :param db_name: Vektör veritabanının adı (varsayılan: "vector-db")
    :param uploaded_files_dir: Kaydedilecek ana dizin (varsayılan: "uploaded_files")
    """
    # PDF'in kaydedileceği klasörü oluştur
    os.makedirs(uploaded_files_dir, exist_ok=True)
    save_path = os.path.join(uploaded_files_dir, os.path.basename(pdf_file.name))  # Güvenli isimlendirme
    
    # PDF dosyasını kaydet
    with open(save_path, "wb") as f:
        f.write(pdf_file.getvalue())  # Gradio dosya nesnesi için doğru yöntem

    print(f"{pdf_file.name} başarıyla kaydedildi!")

    # PDF dosyasını aç
    pdf_open = pymupdf.open(save_path)
    toc = pdf_open.get_toc()
    chunked_documents = []

    # Eğer TOC yoksa, tüm dokümanı tek parça olarak işle
    if not toc:
        chunk_text = ""
        for page_num in range(pdf_open.page_count):
            chunk_text += pdf_open[page_num].get_text()

        chunked_documents.append(
            Document(
                page_content=chunk_text,
                metadata={"heading": "Full Document", "start_page": 0, "end_page": pdf_open.page_count}
            )
        )
    else:
        # TOC kullanarak bölümlere ayır
        for i, item in enumerate(toc):
            heading = item[1]
            start_page = item[2]
            
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = pdf_open.page_count - 1  # Son başlıksa son sayfaya kadar
            
            chunk_text = ""
            for page_num in range(start_page, end_page):
                chunk_text += pdf_open[page_num].get_text()
            
            chunked_documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={"heading": heading, "start_page": start_page, "end_page": end_page}
                )
            )

    # Chunking işlemi (Bölme)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_chunks = text_splitter.split_documents(chunked_documents)

    # Vektör veritabanını güncelle
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embed_method)
    vectorstore.add_documents(documents=final_chunks)
    vectorstore.persist()

    return f"{pdf_file.name} başarıyla vektör veritabanına eklendi!"
