from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

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

retrieval_grader_prompt = PromptTemplate(
    template=RETRIEVAL_GRADER_TEMPLATE,
    input_variables=["question", "documents"],
)
llm = ChatOllama(model="llama3.2", format="json", temperature=0)
retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
