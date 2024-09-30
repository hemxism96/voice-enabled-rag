from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI

class LlmGenerator:
    def __init__(self, vectorstore) -> None:
        self.llm = ChatMistralAI(model="mistral-small")
        self.retriever = vectorstore.as_retriever()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
            ("Question", "{user_input}"),
            
        ])

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate(self):
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )