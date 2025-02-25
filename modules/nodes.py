from modules.agent_state import AgentState
from helpers.config import embeddings, vectorstore, retriever, llm, memory

class Nodes:
    @staticmethod
    def user_input_node(state: AgentState):
        # query alır
        return {"user_query": state["user_query"]}

    @staticmethod
    def retrieve_node(state: AgentState):
        # vector dbden alakalıları çekterim
        retrieved_docs = vectorstore.similarity_search(state["user_query"], k=10)
        return {"retrieved_docs": [doc.page_content for doc in retrieved_docs]}

    @staticmethod
    def generate_response_node(state: AgentState):
        
        # Konuşma geçmişini al
        chat_history = memory.load_memory_variables({}).get("history", "")
        
        # cevabı generate ettiririm
        combined_context = "\n".join(state["retrieved_docs"])
        prompt = f"""
        You are a telecom assistant. Your answers should be based on the context and chat history provided. If the context is not relevant to the user's query, politely state that you do not have the required information.

        Chat History: {chat_history}    
        
        Context: {combined_context}
        
        Question: {state['user_query']}
        
        Answer:
        """
        response = llm.invoke(prompt)
        return {"response": response}

    @staticmethod
    def update_memory_node(state: AgentState):
        # konuşma geçmişini güncellerim
        memory.save_context({"input": state["user_query"]}, {"output": state["response"]})
        return {"chat_history": memory.load_memory_variables({}).get("history", "")}