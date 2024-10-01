from typing_extensions import TypedDict
from typing import List

from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults


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
retrieval_grader_prompt = PromptTemplate(
    template=retrieval_grader_template,
    input_variables=["question", "documents"],
)

generator_template = """You are an assistant for question-answering tasks. 

Use the following documents to answer the question. 

If you don't know the answer, just say that you don't know. 

Use three sentences maximum and keep the answer concise:
Question: {question} 
Documents: {documents} 
Answer: 
"""
generator_prompt = PromptTemplate(
    template=generator_template,
    input_variables=["question", "documents"]
)



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]


def getLanggraph(db, llm):
    retriever = db.as_retriever(k=3)
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
    web_search_tool = TavilySearchResults(k=3)
    rag_chain = generator_prompt | llm | StrOutputParser()
    print(rag_chain)

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]
        documents = retriever.invoke(question)
        steps = state["steps"]
        steps.append("retrieve_documents")
        return {"documents": documents, "question": question, "steps": steps}


    def grade_documents(state):
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
            grade = retrieval_grader.invoke(
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


    def web_search(state):
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
        web_results = web_search_tool.invoke({"query": question})
        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]
        )
        return {"documents": documents, "question": question, "steps": steps}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"documents": documents, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }
    
    
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("web_search", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile()

    return custom_graph