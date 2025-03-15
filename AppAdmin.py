import os
import pymupdf
import shutil
import json
import hashlib  
import gradio as gr
import pandas as pd
from datetime import datetime
from langgraph.graph import StateGraph, END
from modules.agent_state import AgentState
from modules.nodes import Nodes
from helpers.constants import DB_NAME, UPLOADED_FILES_DIR
from helpers.config import vectorstore, embeddings, llm, text_splitter, llm, memory
from helpers.logger import log_event
from modules.document_handle import store_pdf, remove_pdf_from_db
from langchain_community.document_loaders import PyMuPDFLoader

# ------------------------------
# Helper Function to Get PDF Title
# ------------------------------
def get_pdf_title(pdf_name):
    """Retrieve the first title of a PDF given its source name."""
    title_list = vectorstore.get(where={"source": pdf_name})["metadatas"]
    titles = list(set(meta["title"] for meta in title_list if "title" in meta))
    return titles[0] if titles else None

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

# File to store PDF metadata
HASH_FILE = os.path.join(UPLOADED_FILES_DIR, "pdf_metadata.json")

# Load metadata from JSON file
def load_metadata():
    """Load stored metadata from the JSON file."""
    if not os.path.exists(HASH_FILE):
        return {}
    try:
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

# Save metadata to JSON file
def save_metadata(metadata):
    """Save metadata to the JSON file."""
    with open(HASH_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

# Compute the hash value of a PDF file
def compute_pdf_hash(file_path):
    """Compute the SHA-256 hash value of a PDF file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

# Handle PDF upload with metadata
def save_and_process_pdf(pdf_file, description):
    """Save the uploaded PDF with metadata."""
    if pdf_file is None:
        return "Please upload a PDF file."
    
    os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)
    pdf_filename = os.path.basename(pdf_file.name)
    save_path = os.path.join(UPLOADED_FILES_DIR, pdf_filename)
    
    # Compute hash value BEFORE copying the file
    new_pdf_hash = compute_pdf_hash(pdf_file.name)
    
    # Load existing metadata
    metadata = load_metadata()
    
    # If the hash already exists, prevent re-upload
    if new_pdf_hash in metadata:
        return "This PDF has already been uploaded."
    
    # Copy the file AFTER confirming it's unique
    shutil.copy2(pdf_file.name, save_path)
    
    # Get file size in MB
    pdf_size = os.path.getsize(save_path) / (1024 * 1024)
    
    # Get current date
    uploaded_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # Store metadata
    metadata[new_pdf_hash] = {
        "pdf_name": pdf_filename,
        "size": round(pdf_size, 4),  # Rounded to 4 decimal places
        "uploaded_date": uploaded_date,
        "description": description
    }
    save_metadata(metadata)
    
    # Add the PDF to the vector database
    result = store_pdf(save_path)
    return result

# List uploaded PDFs as a DataFrame
def list_uploaded_pdfs_as_df():
    """
    List PDF files in the upload folder as a DataFrame with metadata
    """
    try:
        metadata = load_metadata()
        if metadata:
            pdf_list = [
                {
                    "PDF Name": data["pdf_name"],
                    "Title": get_pdf_title(data["pdf_name"]),  # Add title column
                    "Size (mb)": data["size"],
                    "Uploaded Date": data["uploaded_date"],
                    "Description": data["description"]
                }
                for data in metadata.values()
            ]
             # Sort by Uploaded Date in descending order (newest first)
            df = pd.DataFrame(pdf_list)
            df = df.sort_values(by=["Uploaded Date"], ascending=False)
            return df
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
    """Store the selected PDF name if it's a valid selection."""
    global selected_pdf
    if isinstance(evt.value, str) and evt.value.endswith(".pdf"):
        selected_pdf = evt.value
        return f"Selected: {selected_pdf}"
    selected_pdf = None  # Reset selection if it's not a valid PDF
    return "Invalid selection. Please select a PDF name."

# Handle PDF removal
def handle_pdf_remove():
    """Handle the removal of the selected PDF and delete its hash record."""
    global selected_pdf
    if not selected_pdf or selected_pdf == "No PDFs uploaded":
        return "No PDF selected for removal.", list_uploaded_pdfs_as_df(), get_collection_count()
    
    # Load existing metadata
    metadata = load_metadata()
    
    # Ensure the selected item is a valid PDF name
    if selected_pdf not in [data["pdf_name"] for data in metadata.values()]:
        return "Invalid selection. Please select a valid PDF name.", list_uploaded_pdfs_as_df(), get_collection_count()
    
    # Find the corresponding hash for the selected PDF
    pdf_hash_to_remove = None
    for hash_key, data in metadata.items():
        if data["pdf_name"] == selected_pdf:
            pdf_hash_to_remove = hash_key
            break
    
    if not pdf_hash_to_remove:
        return "PDF not found in metadata.", list_uploaded_pdfs_as_df(), get_collection_count()
    
    # Remove the file from the system
    file_path = os.path.join(UPLOADED_FILES_DIR, selected_pdf)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Remove from metadata
    del metadata[pdf_hash_to_remove]
    save_metadata(metadata)
    
    # Remove from vector DB
    result = remove_pdf_from_db(selected_pdf)

    # Reset selection
    selected_pdf = None

    return result, list_uploaded_pdfs_as_df(), get_collection_count()
# ------------------------------
# GRADIO INTERFACE SETUP
# ------------------------------

with gr.Blocks(title="IZTECH Telecom RAG Server 🤖", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.TabItem("Dashboard"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload PDF and Add to Chroma DB")
                    file_input = gr.File(label="Select a PDF")
                    description_input = gr.Textbox(label="Enter Description")
                    upload_button = gr.Button("Add PDF to Database")
                    output_text = gr.Textbox(label="Upload Result")
                    chunk_count = gr.Textbox(get_collection_count(), interactive=False, label="Total Chunks")
                
                with gr.Column():
                    gr.Markdown("### PDF Management")
                    pdf_dataframe = gr.Dataframe(
                        value=list_uploaded_pdfs_as_df(),
                        datatype=["str"],
                        interactive=False,
                        label="Available PDFs"
                    )
                    selection_text = gr.Textbox(label="Selection Status", value="No data selected.")
                    remove_button = gr.Button("Remove Selected PDF", variant="stop")
                    remove_output_text = gr.Textbox(label="Removal Result")
            
            def update_after_upload(pdf, description):
                result = save_and_process_pdf(pdf, description)
                return result, list_uploaded_pdfs_as_df(), get_collection_count()
            
            # Use button click instead of file change event
            upload_button.click(
                fn=update_after_upload,
                inputs=[file_input, description_input],
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
            chatbot = gr.ChatInterface(fn=respond, title="IZTECH-Server Telecom RAG Assistant 🤖")

if __name__ == "__main__":
    demo.launch(debug=True)