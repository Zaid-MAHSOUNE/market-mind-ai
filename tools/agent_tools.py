import os
from langchain.tools import tool
from tools.stock_tools import get_market_news, get_company_info, get_stock_prices
from tools.rag_storage import get_storage_engine

@tool
def company_entity_resolver(company_name: str):
    """
    Step 1: Use this tool to identify the entity. 
    It searches for the parent company, ownership structure, and stock ticker.
    Crucial for private/state-owned entities like Enedis (EDF), etc.
    """
    # 1. Search for ownership structure first
    search_query = f"Who owns {company_name}? parent company and stock ticker"
    search_results = get_market_news(search_query)
    
    # 2. Ask the LLM to extract the ultimate parent and ticker from search results
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    extraction_prompt = f"""
    Based on these search results for '{company_name}':
    {search_results}
    
    Identify:
    1. The Ultimate Parent Company (e.g., for Enedis, it's EDF).
    2. The most relevant Stock Ticker (e.g., EDF's ticker if Enedis is private).
    3. If the company is French, prioritize Euronext Paris (.PA).
    
    Return a clear summary: "Parent: [Name], Ticker: [Ticker], Status: [Public/Private]"
    """
    
    response = llm.invoke(extraction_prompt)
    return response.content

@tool
def market_data_tool(ticker: str):
    """
    Fetches stock prices and company overview. 
    Use this to check financial health or if a company is delisted.
    """
    info = get_company_info(ticker)
    prices = get_stock_prices(ticker)
    return f"INFO: {info}\nPRICES: {prices}"

@tool
def news_investigator_tool(query: str):
    """
    Searches for latest market news, geopolitical impacts, and market trends.
    """
    return get_market_news(query)

@tool
def internal_knowledge_tool(query: str):
    """
    Consults the internal Knowledge Base (RAG) for PDF reports and strategy documents.
    """
    storage = get_storage_engine()
    return storage.search(query)

# The master list passed to the Agent
tools = [
    company_entity_resolver, 
    market_data_tool, 
    news_investigator_tool, 
    internal_knowledge_tool
]