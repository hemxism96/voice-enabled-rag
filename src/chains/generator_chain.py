import os
from dotenv import load_dotenv

from langchain.prompts.prompt import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

GENERATOR_TEMPLATE = """You are an assistant for question-answering tasks.
Use the following documents to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise:

Question: {question} 
Documents: {documents} 
Answer: 
"""

generator_prompt = PromptTemplate(
        template=GENERATOR_TEMPLATE,
        input_variables=["question", "documents"]
)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    max_length=128,
    temperature=0.5,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_API_KEY,
)
generator_chain = generator_prompt | llm | StrOutputParser()
