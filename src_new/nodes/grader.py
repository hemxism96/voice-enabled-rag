from langchain.prompts.prompt import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser

retrieval_grader_template = """You are a teacher grading a quiz. You will be given: 
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

def get_grader_chain(llm_repo_id, HF_API_KEY):
    retrieval_grader_prompt = PromptTemplate(
        template=retrieval_grader_template,
        input_variables=["question", "documents"],
    )

    llm = HuggingFaceEndpoint(
        repo_id=llm_repo_id,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=HF_API_KEY,
    )

    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

    return retrieval_grader