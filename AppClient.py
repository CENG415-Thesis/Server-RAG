# graph ile beraber CLIENT için yeni yapı kurulacak 
import gradio as gr
from langgraph.graph import StateGraph, END
from modules.agent_state import AgentState
from modules.nodes import Nodes

def build_stategraph():
    builder = StateGraph(AgentState)
    
    # Adding nodes
    builder.add_node("user_input", Nodes.user_input_node)
    builder.add_node("retrieve", Nodes.retrieve_node)
    builder.add_node("generate_response", Nodes.generate_response_node)
    builder.add_node("update_memory", Nodes.update_memory_node)
    
    # Defining flow
    builder.set_entry_point("user_input")
    builder.add_edge("user_input", "retrieve")
    builder.add_edge("retrieve", "generate_response")
    builder.add_edge("generate_response", "update_memory")
    builder.add_edge("update_memory", END)
    
    return builder.compile()

# Create the StateGraph
rag_graph = build_stategraph()

def respond(message, history):
    # Run the graph and get the final state
    final_state = rag_graph.invoke({"user_query": message})
    
    # Extract the AI's response from the final state
    # Adjust this based on your AgentState structure
    ai_response = final_state.get("response", "I don't have an answer for that.")
    
    return ai_response

# Create and launch the Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title="IZTECH Telecom RAG Assistant 🤖",
    theme=gr.themes.Soft()
)


demo.launch(share=True)