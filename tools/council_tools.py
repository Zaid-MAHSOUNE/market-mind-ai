import os
import yfinance as yf
import pandas as pd
import numpy as np
from tavily import TavilyClient
from dotenv import load_dotenv
from crewai.tools import tool

load_dotenv()

# Initialize Tavily once
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# --- 1. THE ANALYST AGENT TOOL ---
@tool("get_analyst_research")
def get_analyst_research(symbol: str, duration: str = "1y"):
    """
    Analyzes fundamental data, SEC filings (10-K/10-Q), and investor transcripts 
    for a given stock symbol. Useful for deep dives into company health.
    Input: symbol (e.g., 'AAPL'), duration (e.g., '1y', '3mo').
    """
    query = f"Detailed fundamental analysis and 10-K/10-Q highlights for {symbol} over the last {duration}"
    
    research = tavily_client.search(
        query=query, 
        search_depth="advanced", 
        max_results=4,
        include_answer=True
    )
    return research

# --- 2. THE SENTIMENT AGENT TOOL ---
@tool("get_sentiment_data")
def get_sentiment_data(symbol: str, duration: str = "7d"):
    """
    Gathers real-time market sentiment, social media buzz, and news headlines 
    for a stock. Use this to gauge investor mood.
    Input: symbol (e.g., 'TSLA'), duration (e.g., '1d', '7d', '1mo').
    """
    # Convert duration labels to integer days for Tavily API
    days_map = {"1d": 1, "7d": 7, "1mo": 30, "3mo": 90}
    search_days = days_map.get(duration, 7)

    return tavily_client.search(
        query=f"market sentiment and social media buzz for {symbol} stock",
        topic="news",
        days=search_days,
        max_results=5
    )

# --- 3. THE RISK AGENT TOOL ---
@tool("get_risk_metrics")
def get_risk_metrics(symbol: str, duration: str = "1y"):
    """
    Calculates technical risk metrics including Annualized Volatility and 
    Value at Risk (VaR) using historical price data from Yahoo Finance.
    Input: symbol (e.g., 'NVDA'), duration (e.g., '1y', '2y').
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=duration)
    
    if hist.empty:
        return f"Error: No historical data found for {symbol} with duration {duration}."

    # Calculate Daily Returns
    returns = hist['Close'].pct_change().dropna()
    
    # 95% Confidence VaR (Historical)
    var_95 = np.percentile(returns, 5)
    volatility = returns.std() * np.sqrt(252) # Annualized Volatility

    return {
        "symbol": symbol,
        "period_analyzed": duration,
        "annualized_volatility": f"{round(volatility * 100, 2)}%",
        "historical_var_95": round(var_95, 4),
        "sector": ticker.info.get("sector", "Unknown")
    }

# --- 4. THE STOCK DATA AGENT ---
@tool("get_stock_technical_data")
def get_stock_technical_data(symbol: str, duration: str = "1mo"):
    """
    Retrieves technical price data (OHLCV) and key financial ratios 
    like P/E, Dividend Yield, and Moving Averages from Yahoo Finance.
    Input: symbol (e.g., 'MSFT'), duration (e.g., '1mo', '1y').
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=duration)
    
    # Key stats from the .info object
    info = ticker.info
    stats = {
        "current_price": info.get("currentPrice"),
        "fifty_day_avg": info.get("fiftyDayAverage"),
        "two_hundred_day_avg": info.get("twoHundredDayAverage"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield", 0)
    }
    
    return {
        "recent_price_action": hist.tail(5).to_dict(), 
        "key_stats": stats
    }