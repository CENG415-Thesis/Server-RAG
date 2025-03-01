from typing import TypedDict, List

# AgentState structure
class AgentState(TypedDict):
    user_query: str
    retrieved_docs: List[str]
    response: str
    chat_history: str
