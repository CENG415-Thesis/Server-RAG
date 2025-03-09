import os
import pymupdf
import gradio as gr
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from helpers.logger import log_event

def store_pdf(pdf_path, db_name="vector-db"):
    """
    Process a PDF file, extract metadata, chunk it, and add to Chroma DB
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pdf_filename = os.path.basename(pdf_path)
    
    try:
        pdf_open = pymupdf.open(pdf_path)
        toc = pdf_open.get_toc()
        pdf_metadata = pdf_open.metadata
        chunked_documents = []
        
        # If no table of contents, process entire document
        if not toc:
            full_text = "".join([page.get_text() for page in pdf_open])
            chunked_documents.append(
                Document(
                    page_content=full_text,
                    metadata={"source": pdf_filename, "heading": "Full Document"}
                )
            )
        else:
            for i, item in enumerate(toc):
                heading = item[1]
                start_page = item[2] - 1  # Adjust to 0-based indexing
                
                if i + 1 < len(toc):
                    end_page = toc[i + 1][2] - 1
                else:
                    end_page = pdf_open.page_count - 1
                
                chunk_text = "".join([pdf_open[page].get_text() for page in range(start_page, end_page + 1)])
                
                dynamic_metadata = {key: value for key, value in pdf_metadata.items() if value}
                
                chunked_documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "source": pdf_filename, 
                            "heading": heading, 
                            "start_page": start_page, 
                            "end_page": end_page, 
                            **dynamic_metadata
                        }
                    )
                )
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_chunks = text_splitter.split_documents(chunked_documents)
        
        # Print debug information
        print(f"Total chunks created: {len(final_chunks)}")
        log_event("ADMIN",f"Total chunks created from external file: {len(final_chunks)}")
        
        # Embedding generation
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            # Test embedding generation
            #test_embedding = embeddings.embed_documents([chunk.page_content for chunk in final_chunks[:1]])
            
            # Chroma vector store
            vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
            
            # Add documents
            vectorstore.add_documents(final_chunks)
            
            print(f"Vectorstore updated with {len(final_chunks)} chunks from {pdf_filename}")
            log_event("ADMIN",f"Vectorstore updated with {len(final_chunks)} chunks from {pdf_filename}")
            
            return f"PDF '{pdf_filename}' successfully processed and added to database."
        
        except Exception as embed_error:
            print(f"Embedding generation error: {embed_error}")
            log_event("ADMIN",f"Embedding generation error: {embed_error}")
            return f"Error generating embeddings: {embed_error}"
    
    except Exception as process_error:
        print(f"PDF processing error: {process_error}")
        log_event("ADMIN",f"PDF processing error: {process_error}")
        return f"Error processing PDF: {process_error}"


def remove_pdf_from_db(pdf_name, db_name="vector-db", upload_folder="uploaded_files_admin"):
    """
    Remove all chunks associated with a given PDF file from the Chroma vector database
    and delete the PDF file from the uploaded files directory.
    
    Args:
        pdf_name (str): Name of the PDF file to remove
        db_name (str, optional): Name of the Chroma vector database directory. Defaults to "vector-db".
        upload_folder (str, optional): Directory containing uploaded PDF files. Defaults to "uploaded_files_admin".
    
    Returns:
        str: Status message of the removal process
    """
    try:
        # Load the existing Chroma vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
        
        # Get all chunk IDs associated with the given PDF file
        chunk_ids = vectorstore.get(where={"source": pdf_name})["ids"]
        
        if not chunk_ids:
            print(f"No chunks found for PDF '{pdf_name}' in the vectorstore.")
            """ 
            # Try to delete the file from the upload directory
            pdf_path = os.path.join(upload_folder, pdf_name)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"Deleted PDF file '{pdf_name}' from {upload_folder}")
                return f"PDF '{pdf_name}' file deleted from storage."
            """
            return f"No data found for PDF '{pdf_name}'."
        
        # Delete the chunks from the vectorstore
        vectorstore.delete(ids=chunk_ids)
        print(f"Successfully removed {len(chunk_ids)} chunks related to '{pdf_name}' from the vectorstore.")
        log_event("ADMIN",f"Successfully removed {len(chunk_ids)} chunks related to '{pdf_name}' from the vectorstore.")
        
        # Delete the PDF file from the upload directory
        pdf_path = os.path.join(upload_folder, pdf_name)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"Deleted PDF file '{pdf_name}' from {upload_folder}")
            log_event("ADMIN",f"Deleted PDF file '{pdf_name}' from {upload_folder}")
        
        return f"PDF '{pdf_name}' successfully removed from database and file storage."
    
    except Exception as e:
        print(f"Error removing PDF '{pdf_name}': {e}")
        log_event("ERROR",f"Error removing PDF '{pdf_name}': {e}" )
        return f"Error: {e}"