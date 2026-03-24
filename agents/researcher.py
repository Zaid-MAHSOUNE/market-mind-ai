import os
from openai import OpenAI
from dotenv import load_dotenv
from tools.rag_storage import MarketMindStorage
from tools.stock_tools import load_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Researcher:
    def __init__(self):
        self.system = load_prompt('researcher_logic.txt')
        self.ticker_resolver_template = load_prompt('ticker_resolver.txt')
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
            model="gpt-4o-mini",
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
        prompt = self.ticker_resolver_template.format(query=query)
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        ticker = completion.choices[0].message.content.strip()
        return ticker