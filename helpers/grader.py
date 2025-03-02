from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
GRADER_PROMPT_TEMPLATE = """
You are an expert evaluator. Grade the response based on the following criteria:
1. Groundedness: Does the response fully address the question using the provided context?
2. Answer Relevance: Is the response relevant and directly answering the question?
3. Context Relevance: Is the response actually correct based on the context?

Context:
{context}

Question:
{question}

Response:
{response}

---

Provide a float score from 0 to 1 for each criteria and a short explanation for the score.
Calculate the average of these criteria and store it in a variable.

For example:
Final Score: 0.2
"""

class Grader:
    def __init__(self):
        self.model = Ollama(model="llama3.2")  # Grading LLM

    def grade(self, context: str, question: str, response: str) -> str:
        prompt_template = ChatPromptTemplate.from_template(GRADER_PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context, question=question, response=response
        )
        return self.model.invoke(prompt)