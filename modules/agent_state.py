from typing import TypedDict, List

# AgentState structure
class AgentState(TypedDict):
    user_query: str
    retrieved_docs: List[str]
    response: str
    chat_history: str
    revision_number: int
    max_revisions: int
    grader_score: float    
