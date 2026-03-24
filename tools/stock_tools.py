import yfinance as yf
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Tavily for Web Search (You'll need an API key from tavily.com)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def get_stock_prices(ticker: str, period: str = "1mo"):
    """Fetches historical price data for a given ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df[['Close', 'Volume']].tail(10).to_string()

def get_market_news(query: str):
    """Searches for the latest financial news and market trends."""
    # This supports the "Investigateur financier" requirement 
    search_result = tavily.search(query=query, search_depth="advanced", max_results=5)
    return search_result

def get_company_info(ticker):
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Safe extraction with defaults
        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "Private/Government Entity")
        
        # Fix the crash here: Check if summary exists before slicing
        summary = info.get("longBusinessSummary")
        if summary:
            summary = summary[:500] + "..."
        else:
            summary = "No public business summary available (Company may be private or nationalized)."
            
        return {
            "name": name,
            "sector": sector,
            "summary": summary
        }
    except Exception as e:
        return {"name": ticker, "sector": "Unknown", "summary": f"Error fetching data: {str(e)}"}
    
def load_prompt(filename):
    path = os.path.join("prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()