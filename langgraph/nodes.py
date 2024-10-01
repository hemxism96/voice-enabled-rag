from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import chromadb

class RAG:
    def __init__(
            self
        ) -> None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
        llm = OllamaLLM(model="llama3.2")
        persistent_client = chromadb.PersistentClient(path="./db/chromadb")
        self.vector_store = Chroma(
            client=persistent_client,
            embedding_function=embeddings,
            collection_name = "rag-chroma"
        )
        self.retriever = self.vector_store.as_retriever(k=2)
        retrieval_grader_prompt = PromptTemplate(
            template="""You are a teacher grading a quiz. You will be given: 
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
            """,
            input_variables=["question", "documents"],
        )
        self.retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
        self.web_search_tool = TavilySearchResults(k=2)

        generator_prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. 
            
            Use the following documents to answer the question. 
            
            If you don't know the answer, just say that you don't know. 
            
            Use three sentences maximum and keep the answer concise:
            Question: {question} 
            Documents: {documents} 
            Answer: 
            """,
            input_variables=["question", "documents"],
        )
        self.rag_chain = generator_prompt | llm | StrOutputParser()


    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]
        documents = self.retriever.invoke(question)
        steps = state["steps"]
        steps.append("retrieve_documents")
        return {"documents": documents, "question": question, "steps": steps}


    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        question = state["question"]
        documents = state["documents"]
        steps = state["steps"]
        steps.append("grade_document_retrieval")
        filtered_docs = []
        search = "No"
        for d in documents:
            grade = self.retrieval_grader.invoke(
                {"question": question, "documents": d.page_content}
            )["score"]
            if grade == "yes":
                filtered_docs.append(d)
            else:
                search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "search": search,
            "steps": steps,
        }
    
    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        search = state["search"]
        if search == "Yes":
            return "search"
        else:
            return "generate"
        
    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        question = state["question"]
        documents = state.get("documents", [])
        steps = state["steps"]
        steps.append("web_search")
        web_results = self.web_search_tool.invoke({"query": question})
        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]
        )
        return {"documents": documents, "question": question, "steps": steps}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"documents": documents, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }