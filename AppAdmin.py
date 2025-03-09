import os
import pymupdf
import shutil
import json
import hashlib  
import gradio as gr
import pandas as pd
from langgraph.graph import StateGraph, END
from modules.agent_state import AgentState
from modules.nodes import Nodes
from helpers.constants import DB_NAME, UPLOADED_FILES_DIR
from helpers.config import vectorstore, embeddings, llm, text_splitter, llm, memory
from helpers.logger import log_event
from modules.document_handle import store_pdf, remove_pdf_from_db
from langchain_community.document_loaders import PyMuPDFLoader

# ------------------------------
# BUILD THE RAG STATEGRAPH PIPELINE
# ------------------------------

def build_stategraph():
    """
    Define and build the StateGraph for the RAG system.
    This graph represents the sequence of steps involved in processing user queries.
    """
    builder = StateGraph(AgentState)
    
    # Adding processing nodes (steps)
    builder.add_node("user_input", Nodes.user_input_node)
    builder.add_node("retrieve", Nodes.retrieve_node)
    builder.add_node("generate_response", Nodes.generate_response_node)
    builder.add_node("update_memory", Nodes.update_memory_node)
    
    # Define the flow of execution
    builder.set_entry_point("user_input")  # First step: process user input
    builder.add_edge("user_input", "retrieve")  # Retrieve relevant chunks
    builder.add_edge("retrieve", "generate_response")  # Generate AI response
    builder.add_edge("generate_response", "update_memory")  # Store conversation memory
    builder.add_edge("update_memory", END)  # End of flow
    
    return builder.compile()

# Create and compile the RAG StateGraph
rag_graph = build_stategraph()

# ------------------------------
# CHATBOT RESPONSE FUNCTION
# ------------------------------

def respond(message, history):
    """
    Process user input and generate a response using the RAG pipeline.
    
    Args:
        message (str): User's query.
        history (list): Conversation history.

    Returns:
        str: AI-generated response.
    """
    # Run the state graph with the user query
    final_state = rag_graph.invoke({"user_query": message})
    
    # Extract the AI's response from the final state
    ai_response = final_state.get("response", "I don't have an answer for that.")
    
    return ai_response

# ------------------------------
# PDF FILE HANDLING FUNCTIONS
# ------------------------------

# File to store PDF hashes
HASH_FILE = os.path.join(UPLOADED_FILES_DIR, "pdf_hashes.json")

# Load hash records from JSON file
def load_hashes():
    """Load stored hash values from the JSON file."""
    if not os.path.exists(HASH_FILE):
        return set()
    try:
        with open(HASH_FILE, "r") as f:
            return set(json.load(f))
    except json.JSONDecodeError:
        return set()

# Save hash records to JSON file
def save_hashes(hashes):
    """Save hash values to the JSON file."""
    with open(HASH_FILE, "w") as f:
        json.dump(list(hashes), f, indent=4)

# Compute the hash value of a PDF file
def compute_pdf_hash(file_path):
    """Compute the SHA-256 hash value of a PDF file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

# Handle PDF upload with hash validation
def save_and_process_pdf(pdf_file):
    """Save the uploaded PDF only if it is not a duplicate."""
    if pdf_file is None:
        return "Please upload a PDF file."

    os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)
    pdf_filename = os.path.basename(pdf_file.name)
    save_path = os.path.join(UPLOADED_FILES_DIR, pdf_filename)

    # Compute hash value BEFORE copying the file
    new_pdf_hash = compute_pdf_hash(pdf_file.name)

    # Load existing hash records
    stored_hashes = load_hashes()

    # If the hash already exists, prevent re-upload
    if new_pdf_hash in stored_hashes:
        return "This PDF has already been uploaded."

    # Copy the file AFTER confirming it's unique
    shutil.copy2(pdf_file.name, save_path)

    # Add the new hash and save it to JSON
    stored_hashes.add(new_pdf_hash)
    save_hashes(stored_hashes)

    # Add the PDF to the vector database
    result = store_pdf(save_path)
    return result

def list_uploaded_pdfs_as_df():
    """
    List PDF files in the upload folder as a DataFrame
    """
    try:
        pdfs = [f for f in os.listdir(UPLOADED_FILES_DIR) if f.lower().endswith('.pdf')]
        if pdfs:
            return pd.DataFrame({"PDF Name": pdfs})
        else:
            return pd.DataFrame({"PDF Name": ["No PDFs uploaded"]})
    except Exception as e:
        return pd.DataFrame({"PDF Name": [f"Error listing PDFs: {e}"]})

def get_collection_count():
    """Get the latest collection count."""
    return f"Total chunks in vector store: {vectorstore._chroma_collection.count()}"

# Store the selected PDF name
selected_pdf = None

def pdf_selection(evt: gr.SelectData):
    """Store the selected PDF name"""
    global selected_pdf
    selected_pdf = evt.value
    return f"Selected: {selected_pdf}"

def handle_pdf_remove():
    """Handle the removal of the selected PDF and delete its hash record."""
    global selected_pdf
    if not selected_pdf or selected_pdf == "No PDFs uploaded":
        return "No PDF selected for removal.", list_uploaded_pdfs_as_df(), get_collection_count()

    # Define the full file path
    file_path = os.path.join(UPLOADED_FILES_DIR, selected_pdf)

    # Compute the hash of the PDF to remove
    pdf_hash = compute_pdf_hash(file_path)

    # Load existing hash records
    stored_hashes = load_hashes()

    # Remove the hash if it exists
    if pdf_hash in stored_hashes:
        stored_hashes.remove(pdf_hash)
        save_hashes(stored_hashes)  # Save the updated hash list

    # Remove the file from the vector database
    result = remove_pdf_from_db(selected_pdf)

    # Reset selection
    selected_pdf = None

    return result, list_uploaded_pdfs_as_df(), get_collection_count()


# ------------------------------
# GRADIO INTERFACE SETUP
# ------------------------------

with gr.Blocks(title="IZTECH Telecom RAG Server 🤖", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.TabItem("PDF Upload"):
            gr.Markdown("### Upload PDF and Add to Chroma DB")
            
            with gr.Row():
                file_input = gr.File(label="Select a PDF")
            
            with gr.Row():    
                upload_button = gr.Button("Add PDF to Database")
            
            output_text = gr.Textbox(label="Upload Result")
            chunk_count = gr.Textbox(get_collection_count(), interactive=False, label="Total Chunks")
            
            gr.Markdown("### PDF Management")
            
            # Define pdf_dataframe first
            pdf_dataframe = gr.Dataframe(
                value=list_uploaded_pdfs_as_df(),
                headers=["PDF Name"],
                datatype=["str"],
                interactive=False,
                label="Available PDFs"
            )
            
            selection_text = gr.Textbox(label="Selection Status", value="No PDF selected")
            remove_button = gr.Button("Remove Selected PDF", variant="stop")
            remove_output_text = gr.Textbox(label="Removal Result")
            
            def update_after_upload(pdf):
                result = save_and_process_pdf(pdf)
                return result, list_uploaded_pdfs_as_df(), get_collection_count()
            
            # Use button click instead of file change event
            upload_button.click(
                fn=update_after_upload,
                inputs=file_input,
                outputs=[output_text, pdf_dataframe, chunk_count]
            )
            
            # Set up event listener for PDF selection - just stores the selection
            pdf_dataframe.select(
                fn=pdf_selection,
                outputs=[selection_text]
            )
            
            # Use button click for PDF removal
            remove_button.click(
                fn=handle_pdf_remove,
                outputs=[remove_output_text, pdf_dataframe, chunk_count]
            )

        with gr.TabItem("Chat"):
            gr.Markdown("### Chat with the Assistant")
            chatbot = gr.ChatInterface(fn=respond, title="IZTECH-Server Telecom RAG Assistant 🤖")

if __name__ == "__main__":
    demo.launch(debug=True)
