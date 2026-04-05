import os
import io
import base64
import json
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Annotated, Dict, Any, Union
from tavily import TavilyClient
from dotenv import load_dotenv

# Load credentials
load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# =================================================================================
# 1. CORE LOGIC FUNCTIONS (The "Engine")
# =================================================================================

def get_detailed_fundamentals(ticker: Annotated[str, "The stock ticker symbol, e.g. AAPL"]) -> str:
    """Calculates P/E, PEG, ROE, Debt-to-Equity, FCF, and Intrinsic Value."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        if income_stmt.empty or balance_sheet.empty:
            return f"Error: Could not retrieve full financial statements for {ticker}."

        rev = income_stmt.loc['Total Revenue'].iloc[0]
        net_inc = income_stmt.loc['Net Income'].iloc[0]
        eps = info.get('trailingEps', 0)
        total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
        equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
        op_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0]
        capex = abs(cash_flow.loc['Capital Expenditure'].iloc[0]) if 'Capital Expenditure' in cash_flow.index else 0
        
        pe_ratio = info.get('trailingPE', "N/A")
        growth_rate = info.get('earningsGrowth', 0) * 100 
        peg_ratio = info.get('pegRatio', "N/A")
        roe = (net_inc / equity) * 100
        debt_to_equity = total_debt / equity
        fcf = op_cash_flow - capex
        
        # Simple Intrinsic Value Calculation
        g = growth_rate if growth_rate > 0 else 1
        intrinsic_value = (eps * (8.5 + 2 * (g/10)) * 4.4) / 4.5

        return f"""
        --- FINANCIAL CHECKLIST: {ticker} ---
        - Revenue: ${rev:,.2f} | Net Income: ${net_inc:,.2f}
        - EPS: ${eps} | Free Cash Flow: ${fcf:,.2f}
        --- RATIOS ---
        - P/E: {pe_ratio} | PEG: {peg_ratio} | ROE: {roe:.2f}% | D/E: {debt_to_equity:.4f}
        --- VALUATION ---
        - Current Price: ${info.get('currentPrice')}
        - Estimated Intrinsic Value: ${intrinsic_value:.2f}
        """
    except Exception as e:
        return f"Error calculating fundamentals for {ticker}: {str(e)}"

def news_investigator_tool(query: Annotated[str, "Search query for news"]) -> str:
    """Deep web search via Tavily for fresh market trends and geopolitical impacts."""
    try:
        search_result = tavily.search(query=query, search_depth="advanced", max_results=5)
        cleaned_results = []
        for r in search_result.get('results', []):
            content = r.get('content', '')
            truncated = content[:1000] + "..." if len(content) > 1000 else content
            cleaned_results.append({"title": r.get('title'), "url": r.get('url'), "content": truncated})
        return json.dumps(cleaned_results)
    except Exception as e:
        return f"News search error: {e}"

def market_data_tool(
    ticker: Annotated[str, "Ticker symbol"], 
    period: Annotated[str, "Lookback period: 1mo, 6mo, 1y, 5y"] = "1y"
) -> str:
    """Fetches historical price stats and sample data for the chosen duration."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty: return f"No data found for {ticker}."
            
        summary = {
            "start_price": hist['Close'].iloc[0],
            "end_price": hist['Close'].iloc[-1],
            "high": hist['High'].max(),
            "low": hist['Low'].min(),
            "avg_volume": hist['Volume'].mean()
        }
        return f"PERIOD ({period}) STATS: {summary}\nSAMPLE:\n{hist[['Close', 'Volume']].tail(5).to_string()}"
    except Exception as e:
        return f"Market data error: {e}"

def get_company_portfolio_breakdown(ticker: Annotated[str, "Ticker symbol"]) -> str:
    """Retrieves business segments, sector, and industry overview."""
    try:
        info = yf.Ticker(ticker).info
        summary = {
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "summary": (info.get("longBusinessSummary")[:500] + "...") if info.get("longBusinessSummary") else "N/A"
        }
        return json.dumps(summary)
    except Exception as e:
        return f"Portfolio fetch error: {e}"

# =================================================================================
# 2. AUTOGEN INTEGRATION LOGIC
# =================================================================================

def register_council_tools(assistant, user_proxy):
    """
    Binds all financial tools to the AutoGen agents.
    'assistant' is the agent that decides to use tools.
    'user_proxy' is the agent that executes them.
    """
    
    # List of functions to register
    tools = [
        get_detailed_fundamentals,
        news_investigator_tool,
        market_data_tool,
        get_company_portfolio_breakdown
    ]

    for tool in tools:
        assistant.register_for_llm(name=tool.__name__, description=tool.__doc__)(tool)
        user_proxy.register_for_execution(name=tool.__name__)(tool)

    print("✅ Financial Council tools registered successfully.")

# =================================================================================
# 3. HELPER FOR UI (Plotting)
# =================================================================================

def create_performance_plot(ticker, amount, period="1y"):
    """Used by the Streamlit UI to show the 'Why' behind a decision."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        initial_price = hist['Close'].iloc[0]
        growth = (hist['Close'] / initial_price) * amount
        
        plt.figure(figsize=(10, 4))
        plt.style.use('dark_background')
        plt.plot(growth, color='#00ffcc', linewidth=2)
        plt.title(f"Historical Growth Simulation: {ticker}", color='white')
        plt.grid(alpha=0.2)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except:
        return None