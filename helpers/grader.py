import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from helpers.config import vectorstore , retriever
from langchain_core.messages import HumanMessage, SystemMessage


local_llm = "llama3.2"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
def get_context_relevance(question, docs):
    """
    Kullanıcının sorusuna verilen yanıtın bağlam uygunluk skorunu hesaplar.

    Parametreler:
        question (str): Kullanıcının sorduğu soru.
        retriever (object): vektör tabanlı geri getirici (vectorstore.as_retriever()).
        llm_json_mode (object): JSON formatında yanıt döndüren LLM modeli.

    Dönüş:
        float: Belgenin soru ile olan bağlam uygunluk skoru (0 ile 1 arasında).
    """

    # Doc grader talimatları
    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

    Give a float score between 0 and 1 depending on the document's relevance to the question."""

    # Grader prompt şablonu
    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

    Carefully and objectively assess whether the document contains at least some information that is relevant to the question.

    Return JSON with two keys, relevance_score, that allows the document to be scaled between 0 and 1 according to the question asked.And a key, explanation_context_relevance, that contains an explanation of the score."""
  

    # Promptu formatla
    doc_grader_prompt_formatted = doc_grader_prompt.format(
        document=docs[0], question=question
    )

    # LLM'e gönder
    result = llm_json_mode.invoke(
        [SystemMessage(content=doc_grader_instructions)]
        + [HumanMessage(content=doc_grader_prompt_formatted)]
    )

    # JSON parse edip relevance score'u döndür
    context_relevance_dict = json.loads(result.content)
    return context_relevance_dict




def get_hallucination_score(docs, generated_content, llm_json_mode):
    """
    Evaluates the hallucination score of a generated response based on retrieved documents.

    Parameters:
        docs (list): A list of retrieved documents.
        generated_content (str): The generated response to be evaluated.
        llm_json_mode (object): A language model that returns JSON-formatted responses.

    Returns:
        float: The hallucination score (0 to 1), where 1 means fully grounded and 0 means not grounded at all.
    """

    # Hallucination grader instructions
    hallucination_grader_instructions = """
    You are a teacher grading a quiz. 

    You will be given FACTS and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score. 

    A score of 0 means that the student's answer does not meet any of the criteria. This is the lowest possible score.

    A score between 0 and 1 reflects partial correctness, where the answer is somewhat grounded in the FACTS.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

    # Grader prompt template
    hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 
    Return JSON with two keys,hallucination_score, floating-point number between 0 and 1 to indicate how well the STUDENT ANSWER is grounded in the FACTS. A score of 1 means the answer is fully grounded, while a score of 0 means it is not grounded at all. And a key, explanation_hallucination, that contains an explanation of the score."""

    
    # Format retrieved documents

    # Format the prompt
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=docs[0], generation=generated_content
    )

    # Invoke the LLM for hallucination grading
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )

    # Parse JSON output
    hallucination_dict = json.loads(result.content)

    return hallucination_dict  

# Example usage:
# hallucination_score = get_hallucination_score(docs, generated_content, llm_json_mode)
# print(hallucination_score)
