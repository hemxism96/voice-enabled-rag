from sentence_transformers import CrossEncoder


def get_reranker() -> CrossEncoder:
    """
    Initializes and returns a CrossEncoder model for reranking.

    The CrossEncoder model is used to rerank a list of documents or answers 
    based on their relevance to a query. This implementation loads the 
    'mixedbread-ai/mxbai-rerank-large-v1' model from Hugging Face, which 
    is fine-tuned for document retrieval and reranking tasks.

    Returns:
        CrossEncoder: The CrossEncoder model for reranking.
    """
    reranker = CrossEncoder(model_name="mixedbread-ai/mxbai-rerank-large-v1")
    return reranker