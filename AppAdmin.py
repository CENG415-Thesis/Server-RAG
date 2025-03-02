import gradio as gr
from langgraph.graph import StateGraph, END
from modules.agent_state import AgentState
from modules.nodes import Nodes
from modules.adminNodes import AdminNodes
from modules.document_handler import save_uploaded_files, list_uploaded_files, remove_pdf_from_vectorstore
from helpers.constants import UPLOADED_FILES_DIR
from helpers.config import vectorstore


def build_server_stategraph():
    """Build the StateGraph for the server"""
    builder = StateGraph(AgentState)
    
    # Add server-specific nodes
    builder.add_node("load_pdf", AdminNodes.load_pdf_node)
    builder.add_node("update_vector_store", AdminNodes.update_vector_store_node)
    
    # Add shared nodes (for chat functionality)
    builder.add_node("user_input", Nodes.user_input_node)
    builder.add_node("retrieve", Nodes.retrieve_node)
    builder.add_node("generate_response", Nodes.generate_response_node)
    builder.add_node("update_memory", Nodes.update_memory_node)
    
    # Define PDF processing flow
    builder.add_conditional_edges(
        "user_input",
        lambda state: "load_pdf" if state.get("pdf_files") else "retrieve"
    )
    builder.add_edge("load_pdf", "update_vector_store")
    builder.add_edge("update_vector_store", END)
    
    # Define chat flow
    builder.add_edge("retrieve", "generate_response")
    builder.add_edge("generate_response", "update_memory")
    builder.add_edge("update_memory", END)
    
    # Set entry point
    builder.set_entry_point("user_input")
    
    return builder.compile()

# Create a separate graph for PDF processing to avoid state validation issues
def build_pdf_processing_graph():
    """Build a separate StateGraph just for PDF processing"""
    # Create a custom state class for PDF processing that doesn't require the same fields
    class PDFProcessState(dict):
        pass
    
    builder = StateGraph(PDFProcessState)
    
    # Add PDF processing nodes
    builder.add_node("load_pdf", AdminNodes.load_pdf_node)
    builder.add_node("update_vector_store", AdminNodes.update_vector_store_node)
    
    # Define flow
    builder.set_entry_point("load_pdf")
    builder.add_edge("load_pdf", "update_vector_store")
    builder.add_edge("update_vector_store", END)
    
    return builder.compile()

# Create the StateGraphs
server_graph = build_server_stategraph()
pdf_process_graph = build_pdf_processing_graph()


def process_pdfs(pdf_files):
    """Process uploaded PDF files and update the vector store"""
    if not pdf_files:
        return "No PDF files uploaded"
    
    # Save the uploaded files to the upload directory, checking for duplicates
    saved_paths, duplicate_files = save_uploaded_files(pdf_files)
    
    # Handle case where all files are duplicates
    if not saved_paths and duplicate_files:
        return f"All files are duplicates and were not uploaded again: {', '.join(duplicate_files)}"
    
    # Handle case with some duplicates
    duplicate_message = ""
    if duplicate_files:
        duplicate_message = f"\nSkipped duplicate files: {', '.join(duplicate_files)}"
    
    # If no new files after duplicate check
    if not saved_paths:
        return f"No new files to process.{duplicate_message}"
    
    try:
        # Use the dedicated PDF processing graph instead of the main server graph
        final_state = pdf_process_graph.invoke({
            "pdf_files": saved_paths
        })
        
        status = final_state.get("status", "Processing completed")
        return f"{status}\nFiles saved to {UPLOADED_FILES_DIR} directory.{duplicate_message}"
    except Exception as e:
        return f"Error processing PDFs: {str(e)}\nFiles were saved to {UPLOADED_FILES_DIR} directory.{duplicate_message}"


def respond(message, history):
    """Handle chat interaction"""
    # Run the graph with user query
    final_state = server_graph.invoke({
        "user_query": message
    })
    
    # Extract the AI's response
    ai_response = final_state.get("response", "I don't have an answer for that.")
    
    return ai_response
def get_collection_count():
    return vectorstore._chroma_collection.count()
    
# Create Gradio interface with tabs for different functions
with gr.Blocks(title="IZTECH Telecom RAG Server 🤖", theme=gr.themes.Soft()) as demo:   
    with gr.Tabs():
        # PDF Upload Tab
        with gr.TabItem("PDF Upload"):
            
            with gr.Row():
                gr.Markdown("Upload PDF files to update the vector database with new documents. Duplicate files will be skipped.")
            
            with gr.Row():
                pdf_input = gr.File(
                    file_count="multiple", 
                    file_types=['.pdf'], 
                    label="Upload PDF Files"
                )
            collection_num =get_collection_count()
            gr.Textbox(f"Total documents in vector store: {collection_num}", interactive=True)
                
            with gr.Row():
                process_btn = gr.Button("ADD to Vector DB")
                status_output = gr.Textbox(label="Processing Status", interactive=False)
            
            with gr.Row(elem_classes="center-row"):
                files_list = gr.DataFrame(
                    headers=["Filename", "Size (KB)", "Upload Date"],
                    datatype=["str", "str", "str"],
                    label="Uploaded PDFs",
                    interactive=False
                )

            # Connect process button to process_pdfs function
            process_btn.click(
                fn=process_pdfs,
                inputs=[pdf_input],
                outputs=[status_output]
            ).then(
                fn=list_uploaded_files,
                outputs=[files_list]
            )

            # Initialize the files list on page load
            demo.load(
                fn=list_uploaded_files,
                outputs=[files_list]
            )
        
        # Chat Tab
        with gr.TabItem("Chat"):
            chatbot = gr.ChatInterface(
                fn=respond,
                title="IZTECH-Server Telecom RAG Assistant 🤖",
                #add min size
                fill_height=True,
            )
            
            
demo.launch(share=True)