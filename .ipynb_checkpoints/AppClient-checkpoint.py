# graph ile beraber CLIENT için yeni yapı kurulacak 
import gradio as gr
from langgraph.graph import StateGraph, END
from modules.agent_state import AgentState
from modules.nodes import Nodes
THRESHOLD_SCORE = 0.7

def build_stategraph():
    builder = StateGraph(AgentState)
    
    # Adding nodes
    builder.add_node("user_input", Nodes.user_input_node)
    builder.add_node("retrieve", Nodes.retrieve_node)
    builder.add_node("generate_response", Nodes.generate_response_node)
    builder.add_node("evaluate_response", Nodes.evaluate_response_node)
    builder.add_node("revise_response", Nodes.revise_response_node)
    builder.add_node("update_memory", Nodes.update_memory_node)
    
    # Defining flow
    builder.set_entry_point("user_input")
    builder.add_edge("user_input", "retrieve")
    builder.add_edge("retrieve", "generate_response")
    builder.add_edge("generate_response", "evaluate_response")
    builder.add_conditional_edges(
        "evaluate_response", 
        lambda state: "revise_response" if state["evaluation_score"] < THRESHOLD_SCORE else "update_memory",
        {"revise_response": "revise_response", "update_memory": "update_memory"}
    )

    builder.add_edge("revise_response", "generate_response")
    builder.add_edge("update_memory", END)
    return builder.compile()

# Create the StateGraph
rag_graph = build_stategraph()

def chatbot_interface(user_query):
    state = {"user_query": user_query}
    
    # RAG Graph akışını çalıştır
    output = rag_graph.invoke(state)

    # Eğer hata varsa kullanıcıya göster
    if "error" in output:
        return output["error"]
    
    return output["response"]

# Gradio Arayüzü
iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="Telecom RAG Assistant",
    description="Telecom, kod ve özetleme destekli RAG tabanlı asistan",
    theme="default"
)

# Çalıştır
iface.launch(share=True)


