import os
from dotenv import load_dotenv
from tools.stock_tools import get_stock_prices, get_market_news, get_company_info

# 1. Load Environment Variables
load_dotenv()

def test_financial_tools():
    print("🚀 Starting Tool Integration Test...\n")
    
    # Check if API Keys are present
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY not found in .env file.")
        return

    test_ticker = "NVDA" # Nvidia as a trending example

    # Test 1: Yahoo Finance Basic Info
    print(f"--- Testing Company Info for {test_ticker} ---")
    try:
        info = get_company_info(test_ticker)
        print(f"✅ Success! Company Name: {info['name']}")
        print(f"Sector: {info['sector']}\n")
    except Exception as e:
        print(f"❌ Failed Company Info: {e}")

    # Test 2: Historical Prices (Reasoning Input)
    print(f"--- Testing Price Data (Last 10 Days) ---")
    try:
        prices = get_stock_prices(test_ticker)
        print(f"✅ Success! Data received:\n{prices}\n")
    except Exception as e:
        print(f"❌ Failed Price Data: {e}")

    # Test 3: Tavily Search (ReAct Action)
    print(f"--- Testing Market News Search ---")
    try:
        news = get_market_news(f"Latest stock market trends for {test_ticker}")
        if news and 'results' in news:
            print(f"✅ Success! Found {len(news['results'])} news articles.")
            print(f"Top Headline: {news['results'][0]['title']}\n")
        else:
            print("⚠️ Search returned no results.")
    except Exception as e:
        print(f"❌ Failed Market News Search: {e}")

if __name__ == "__main__":
    test_financial_tools()