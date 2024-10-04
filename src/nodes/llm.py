from langchain.prompts.prompt import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSerializable


GENERATOR_TEMPLATE = """You are an assistant for question-answering tasks.
Use the following documents to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise:

Question: {question} 
Documents: {documents} 
Answer: 
"""


def get_generator_chain(llm_repo_id: str, hf_api_key: str) -> RunnableSerializable:
    """
    Creates a generator chain for processing questions and documents using a specified 
    Hugging Face model.

    This function constructs a generator chain that takes a question and a list of 
    documents, processes them through a prompt template, and utilizes a specified 
    language model hosted on Hugging Face to generate responses.

    Args:
        llm_repo_id (str): The repository ID of the Hugging Face model to use for generation.
        HF_API_KEY (str): The API key for accessing the Hugging Face Hub.

    Returns:
        RunnableSerializable: A chain that processes input through a prompt template, 
                            the language model, and a string output parser.
    """
    generator_prompt = PromptTemplate(
        template=GENERATOR_TEMPLATE,
        input_variables=["question", "documents"]
    )
    llm = HuggingFaceEndpoint(
        repo_id=llm_repo_id,
        max_length=128,
        temperature=0.5,
        repetition_penalty=1.03,
        huggingfacehub_api_token=hf_api_key,
    )
    generator_chain = generator_prompt | llm | StrOutputParser()
    return generator_chain
