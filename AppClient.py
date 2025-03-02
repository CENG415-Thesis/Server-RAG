import gradio as gr
from langgraph.graph import StateGraph, END
from modules.agent_state import AgentState
from modules.nodes import Nodes
from helpers.logger import log_event

THRESHOLD_SCORE = 0.7

def should_continue(state):
        if state["grader_score"] >= THRESHOLD_SCORE:
            log_event("INFO", "Response meets quality threshold. Proceeding to memory update.")
            return "update_memory"
        
        if state["revision_number"] >= state["max_revisions"]:
            log_event("INFO", "Max revisions reached. Proceeding with final response.")
            return "update_memory"
        
        log_event("INFO", "Response needs improvement. Proceeding to revision.")
        return "revise_response"

def build_stategraph():
    builder = StateGraph(AgentState)

    # Düğümleri ekle
    builder.add_node("user_input", Nodes.user_input_node)
    builder.add_node("retrieve", Nodes.retrieve_node)
    builder.add_node("generate_response", Nodes.generate_response_node)
    builder.add_node("evaluate_response", Nodes.evaluate_response_node)
    builder.add_node("revise_response", Nodes.revise_response_node)
    builder.add_node("update_memory", Nodes.update_memory_node)

    # Başlangıç noktası
    builder.set_entry_point("user_input")

    # Akış Bağlantıları
    builder.add_conditional_edges(
        "evaluate_response",
        should_continue,
        {
            "update_memory": "update_memory",
            "revise_response": "revise_response"
        }
    )
    
    builder.add_edge("user_input", "retrieve")
    builder.add_edge("retrieve", "generate_response")
    builder.add_edge("generate_response", "evaluate_response")
    
    builder.add_edge("revise_response", "evaluate_response")
    builder.add_edge("update_memory", END)
    return builder.compile()

rag_graph = build_stategraph()

def respond(message, history):
    final_state = rag_graph.invoke({
        "user_query": message, 
        "max_revisions": 2,
        "revision_number": 1,})
    ai_response = final_state.get("response", "I don't have an answer for that.")
    return ai_response

demo = gr.ChatInterface(
    fn=respond,
    title="IZTECH Telecom RAG Assistant 🤖",
    theme=gr.themes.Soft()
)

demo.launch(share=True)