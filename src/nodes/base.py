from langchain.schema import Document


class Nodes:
    """
    A class to manage various operations related to nodes, including re-querying, 
    retrieving documents, grading relevance, web searching, and generating answers.
    """
    def __init__(self, helpers) -> None:
        """
        Initializes the Nodes class with the provided NodeHelpers instance.

        Args:
            helpers (NodeHelpers): An instance of the NodeHelpers class that provides
            various tools for retrieval, rephrasing, grading, searching, and generation.
        """
        self.retriever = helpers.retriever
        self.reranker = helpers.reranker
        self.retrieval_grader = helpers.retrieval_grader
        self.web_search_tool = helpers.web_search_tool
        self.generator = helpers.generator


    def retrieve(self, state: dict) -> dict:
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
        new_state = {"documents": documents, "question": question, "steps": steps}
        return new_state


    def rerank(self, state: dict) -> dict:
        """
        Rerank documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains best retrieved documents
        """
        question = state["question"]
        documents = [d.page_content for d in state["documents"]]
        reranked_document = self.reranker.rank(
            question, documents, return_documents=True, top_k=3
        )
        reranked_document = [
            Document(page_content=d["text"], metadata={"source": "rerank_doc"})
            for d in reranked_document
        ]
        steps = state["steps"]
        steps.append("rerank")
        new_state = {"documents": reranked_document, "question": question, "steps": steps}
        return new_state


    def grade_documents(self, state: dict) -> dict:
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
        if len(filtered_docs)==0:
            search = "Yes"
        new_state = {
            "documents": filtered_docs,
            "question": question,
            "search": search,
            "steps": steps,
        }
        return new_state


    def decide_to_generate(self, state: dict) -> str:
        """
        Determines whether to generate an answer, or do web research.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        search = state["search"]
        if search == "Yes":
            next_node = "search"
        else:
            next_node = "generate"
        return next_node


    def web_search(self, state: dict) -> dict:
        """
        Web search based on the question.

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
        new_state = {"documents": documents, "question": question, "steps": steps}
        return new_state


    def generate(self, state: dict) -> dict:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        question = state["question"]
        documents = "\n\n".join(doc.page_content for doc in state["documents"])
        generation = self.generator.invoke({"documents": documents, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")
        new_state = {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }
        print(new_state)
        return new_state
