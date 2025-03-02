from modules.agent_state import AgentState
from helpers.config import embeddings, vectorstore, retriever, llm, memory, text_splitter
from langchain_community.document_loaders import PyPDFLoader
from helpers.constants import DB_NAME
import os
from langchain_community.vectorstores import Chroma

class AdminNodes:
    """Admin-specific nodes for the StateGraph"""
    @staticmethod
    def load_pdf_node(state: AgentState):
        """Node to load and process PDF files"""
        pdf_files = state.get("pdf_files", [])
        if not pdf_files:
            return {"status": "No PDF files provided", **state}
        
        all_documents = []
        # Load each PDF
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                return {
                    "status": f"Error loading PDF {pdf_file}: {str(e)}", 
                    # Add a placeholder response to satisfy the state requirements
                    "response": f"Error loading PDF: {str(e)}",
                    **state
                }
        
        split_docs = text_splitter.split_documents(all_documents)
        
        return {
            "status": f"Successfully loaded {len(pdf_files)} PDF(s) with {len(split_docs)} chunks",
            "documents": split_docs,
            # Add a placeholder response to satisfy the state requirements
            "response": f"Successfully loaded {len(pdf_files)} PDF(s) with {len(split_docs)} chunks",
            **state
        }
    
    @staticmethod
    def update_vector_store_node(state: AgentState):
        """Node to update the vector store with new documents"""
        documents = state.get("documents", [])
        if not documents:
            return {
                "status": "No documents to add to vector store.", 
                # Add a placeholder response to satisfy the state requirements
                "response": "No documents to add to vector store.",
                **state
            }
        
        try:
            # Check if vector store exists
            if os.path.exists(DB_NAME):
                # Add new documents
                vectorstore.add_documents(documents)
            else:
                # Create new vector store
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=DB_NAME
                )
            
            # Persist changes
            vectorstore.persist()
            
            success_message = f"Successfully updated vector store with {len(documents)} documents!"
            return {
                "status": success_message,
                "vector_store": vectorstore,
                # Add a placeholder response to satisfy the state requirements
                "response": success_message,
                **state
            }
        except Exception as e:
            error_message = f"Error updating vector store: {str(e)}"
            return {
                "status": error_message,
                # Add a placeholder response to satisfy the state requirements
                "response": error_message,
                **state
            }