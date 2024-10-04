from langchain_community.tools.tavily_search import TavilySearchResults


def get_web_tool() -> TavilySearchResults:
    """
    Creates a web search tool for retrieving search results.

    This function initializes an instance of the TavilySearchResults class with a specified 
    number of search results to return (k). In this case, it is set to return 3 results.

    Returns:
        TavilySearchResults: An instance of the TavilySearchResults class configured to return 
                            the specified number of search results.
    """
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool