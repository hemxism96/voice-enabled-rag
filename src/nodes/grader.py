from typing import Optional

from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.base import RunnableSerializable


RETRIEVAL_GRADER_TEMPLATE = """You are a teacher grading a quiz. You will be given:
1/ a QUESTION
2/ A FACT provided by the student

You are grading RELEVANCE RECALL:
A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
1 is the highest (best) score. 0 is the lowest score you can give. 

Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Question: {question} \n
Fact: \n\n {documents} \n\n

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
"""


def get_grader_chain(model_name: Optional[str] = "llama3.2") -> RunnableSerializable:
    """
    Creates a chain for grading retrieved documents based on a given model.

    This function constructs a retrieval grader chain that takes a question and a list of 
    documents, processes them through a prompt template, and utilizes a specified language 
    model to evaluate the relevance of the documents.

    Args:
        model_name (str): The name of the language model to use for grading. 
                        Defaults to "llama3.2".

    Returns:
        LLMChain: A chain that processes input through a prompt template, the language model, 
                and a JSON output parser.
    """
    retrieval_grader_prompt = PromptTemplate(
        template=RETRIEVAL_GRADER_TEMPLATE,
        input_variables=["question", "documents"],
    )
    print(model_name)
    llm = ChatOllama(model=model_name, format="json", temperature=0)
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
    return retrieval_grader
