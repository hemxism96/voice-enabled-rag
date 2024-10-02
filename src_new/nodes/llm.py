from langchain.prompts.prompt import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser


generator_template = """You are an assistant for question-answering tasks. 

Use the following documents to answer the question. 

If you don't know the answer, just say that you don't know. 

Use three sentences maximum and keep the answer concise:
Question: {question} 
Documents: {documents} 
Answer: 
"""

def get_generator_chain(llm_repo_id, HF_API_KEY):
    generator_prompt = PromptTemplate(
        template=generator_template,
        input_variables=["question", "documents"]
    )
    llm = HuggingFaceEndpoint(
        repo_id=llm_repo_id,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=HF_API_KEY,
    )
    generator_chain = generator_prompt | llm | StrOutputParser()
    return generator_chain