from modules.agent_state import AgentState
from helpers.config import embeddings, vectorstore, retriever, llm, memory
from langchain_community.llms import Ollama
from helpers.logger import log_event
from helpers.grader import Grader

THRESHOLD_SCORE = 0.7

class ModelHandler:
    def __init__(self):
        self.models = {
            "telecom": Ollama(model="llama3.2"),
            "code": Ollama(model="codellama"),
            "summary": Ollama(model="mistral")
        }
        self.classifier = Ollama(model="llama3.2")  # LLM Model for classifying

    def classify_query(self, query: str) -> str:
        """Use LLM to classify the query into 'telecom', 'code', or 'summary'."""
        classification_prompt = f"""
        Given the following user query, classify it into one of the following categories:
        - "telecom" if it is related to telecommunications, networking, 5G, or telecom industry topics.
        - "code" if it is related to programming, coding, functions, scripts, or software development.
        - "summary" if it asks for summarization, explanation, or TL;DR-style content.

        Query: "{query}"

        Respond with only one word: 'telecom', 'code', or 'summary'.
        """

        response = self.classifier.generate([classification_prompt])
        category = response.generations[0][0].text.strip().lower() 
        
        # Güvenlik kontrolü: Yanlış cevap dönerse default olarak telecom seç
        if category not in self.models:
            category = "telecom"
            log_event("INFO", f"Classified query: {query} as category: {category}")
        return category


    
    def get_model(self, query: str):
        """Choose the best model based on LLM classification of the query."""
        category = self.classify_query(query)
        log_event("INFO", f"Selected model for category {category}")
        return self.models[category]


class Nodes:
    @staticmethod
    def user_input_node(state: AgentState):
        log_event("INFO", f"User input node started with state: {state}")
        # query alır
        #helpers.logger.log_event("INFO", f"User query received: {state["user_query"]}")
        return {"user_query": state["user_query"]}

    @staticmethod
    def retrieve_node(state: AgentState):
        # vector dbden alakalıları çekterim
        retrieved_docs = vectorstore.similarity_search(state["user_query"], k=10)
        log_event("INFO", f"Retrieved {len(retrieved_docs)} documents")
        return {"retrieved_docs": [doc.page_content for doc in retrieved_docs]}

    @staticmethod
    def generate_response_node(state: AgentState):
        
        model_handler = ModelHandler()

        # En uygun modeli seç
        model = model_handler.get_model(state["user_query"])
        # Konuşma geçmişini al
        chat_history = memory.load_memory_variables({}).get("history", "")
        
        # cevabı generate ettiririm
        combined_context = "\n".join(state["retrieved_docs"])
        prompt = f"""
        You are a telecom assistant. Your answers should be based on the context and chat history provided. If the context is not relevant to the user's query, politely state that you do not have the required information. \
        If there is no chat history, simply say, "Hello, how can I help you?"

        Chat History: {chat_history}    
        
        Context: {combined_context}
        
        Question: {state['user_query']}
        
        Answer:
        """
        try:
            response = model.invoke(prompt)
            log_event("INFO", f"Query: {state['user_query']} | Response: {response}")
            return {"response": response}
        except Exception as e:
            log_event("ERROR", f"Error while generating: {str(e)}")
            return {"error":  "Response not generated"}

    
    @staticmethod
    def evaluate_response_node(state: AgentState):
        grader = Grader()
        log_event("INFO", f"Evaluating response for query: {state['user_query']}")
        evaluation_result = grader.grade(
            context="\n".join(state["retrieved_docs"]),
            question=state["user_query"],
            response=state["response"])

        log_event("INFO", f"Evaluation result: {evaluation_result}")
        print("Grader Output:", evaluation_result)  # Debug için çıktıyı yazdır
    
        scores = {"Groundedness": 0.0, "Answer Relevance": 0.0, "Context Relevance": 0.0}
    
        score_lines = evaluation_result.split("\n")
        for line in score_lines:
            for key in scores.keys():
                if key in line:
                    try:
                        score = float(line.split(":")[1].strip().replace('*', ''))
                        scores[key] = score
                    except (IndexError, ValueError):
                        continue
    
        final_score = sum(scores.values()) / len(scores)  # Üç kriterin ortalamasını al
        log_event("INFO", f"Final evaluation score: {final_score}")
        return {"grader_score": final_score}

    @staticmethod
    def revise_response_node(state: AgentState):
        log_event("INFO", f"Revising response for query: {state['user_query']}")
    
        model_handler = ModelHandler()
        model = model_handler.get_model(state["user_query"])
        
        prompt = f"""
        You are improving an telecom assistant's response based on evaluation feedback. Consider the retrieved context and previous response. 
        Improve clarity, completeness, and relevance to the user's question.

        - User Query: {state['user_query']}
        - Retrieved Context: {" ".join(state["retrieved_docs"])}
        - Previous Response: {state['response']}
        - Evaluation Feedback: {state['grader_score']} (low score means response needs significant improvement)
        
        Please provide an improved version of the response.
        """

        try:
            revised_response = model.invoke(prompt)
            log_event("INFO", f"Revised response generated.")
            return {"response": revised_response, "revision_number": state["revision_number"] + 1}
        except Exception as e:
            log_event("ERROR", f"Error in revising response: {str(e)}")
            return {"error": "Failed to revise response"}
            
    @staticmethod
    def update_memory_node(state: AgentState):
        log_event("INFO", f"Saving memory for query: {state['user_query']} and response: {state['response']}")
        memory.save_context({"input": state["user_query"]}, {"output": state["response"]})
        log_event("INFO", f"Updated memory. Current chat history: {memory.load_memory_variables({}).get('history')}")
        return {"chat_history": memory.load_memory_variables({}).get("history", "")}

