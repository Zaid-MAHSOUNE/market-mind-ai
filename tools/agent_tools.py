import os
from langchain.tools import tool
from tools.stock_tools import get_market_news, get_company_info, get_stock_prices
from tools.rag_storage import get_storage_engine

@tool
def company_identifier_tool(company_name: str):
    """
    Converts a company name to a stock ticker. 
    Handles French companies (.PA) and identifies parent companies for private entities.
    """
    # Note: Import inside function to avoid circular dependency with Researcher
    from agents.researcher import Researcher
    return Researcher().resolve_ticker(company_name)

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
    company_identifier_tool, 
    market_data_tool, 
    news_investigator_tool, 
    internal_knowledge_tool
]