import os
from openai import OpenAI
from dotenv import load_dotenv
from tools.stock_tools import get_market_news, get_company_info, get_stock_prices

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Researcher:
    def __init__(self):
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'researcher_logic.txt')
        with open(prompt_path, 'r') as f:
            self.system = f.read()
        self.messages = [{"role": "system", "content": self.system}]

    def execute_stream(self, message):
        """Returns a generator for real-time streaming of thoughts."""
        self.messages.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=self.messages,
            stream=True  # Enables streaming
        )
        
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content # Yielding chunks for Streamlit
        
        self.messages.append({"role": "assistant", "content": full_response})

    def resolve_ticker(self, query):
        """
        New Method: Converts a company name to a stock ticker.
        If the user provides 'Enedis', it finds the relevant ticker or parent company.
        """
        # Quick internal 'thought' to resolve ticker
        prompt = f"""
        Find the stock ticker symbol for the company: {query}. 
        
        CRITICAL INSTRUCTIONS:
        1. If the company is French, prioritize the Euronext Paris ticker (suffix '.PA').
        2. Example: 'LVMH' -> 'MC.PA', 'Total' -> 'TTE.PA'.
        3. If the company is private (like Enedis), identify the parent company (EDF) and use its last known ticker or specify it is a state-owned entity.
        
        Return ONLY the ticker symbol.
        """
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        ticker = completion.choices[0].message.content.strip()
        return ticker

    def get_raw_data(self, user_input):
        ticker = self.resolve_ticker(user_input)
        
        # 1. Try to get Info (Robustly)
        info = get_company_info(ticker)
        
        # 2. Try to get Prices (Handle delisted case)
        prices = get_stock_prices(ticker)
        if "No data found" in str(prices):
            prices = "STOCK DATA UNAVAILABLE: This entity is likely private or nationalized (e.g., EDF/Enedis)."
        
        # 3. Always get News (This is our fallback for French companies)
        search_query = f"French energy market news {user_input} {ticker} March 2026 stability and geopolitical impact"
        news = get_market_news(search_query)
        
        return f"RESOLVED TICKER: {ticker}\n\nINFO: {info}\n\nMARKET DATA: {prices}\n\nGEOPOLITICAL NEWS: {news}"