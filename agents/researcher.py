import os
from openai import OpenAI
from dotenv import load_dotenv
from tools.stock_tools import get_market_news, get_company_info, get_stock_prices
from tools.rag_storage import MarketMindStorage

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Researcher:
    def __init__(self):
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'researcher_logic.txt')
        with open(prompt_path, 'r') as f:
            self.system = f.read()
        self.messages = [{"role": "system", "content": self.system}]

        self.rag_enabled = False
        # Use the absolute path logic we fixed earlier
        chroma_path = os.path.abspath(os.path.join("data", "chroma_db"))
        
        # If the DB folder exists and isn't empty, just enable it
        if os.path.exists(chroma_path) and any(os.scandir(chroma_path)):
            try:
                self.rag_engine = MarketMindStorage()
                self.rag_enabled = True
            except Exception as e:
                print(f"⚠️ Researcher RAG link failed: {e}")

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