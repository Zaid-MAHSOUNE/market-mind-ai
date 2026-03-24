import os
from openai import OpenAI
from dotenv import load_dotenv
from tools.rag_storage import MarketMindStorage
from tools.stock_tools import load_prompt

# Load environment variables (including the OpenAI API key)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Researcher:
    def __init__(self):
        # Fetch base instructions (the researcher's "brain")
        self.system = load_prompt('researcher_logic.txt')
        self.ticker_resolver_template = load_prompt('ticker_resolver.txt')
        
        # Initialize conversation history with the system message
        self.messages = [{"role": "system", "content": self.system}]

        # RAG is disabled by default
        self.rag_enabled = False
        
        # Point to the directory where the ChromaDB database is stored
        chroma_path = os.path.abspath(os.path.join("data", "chroma_db"))
        
        # If the DB folder exists and isn't empty, try to connect the RAG engine
        if os.path.exists(chroma_path) and any(os.scandir(chroma_path)):
            try:
                self.rag_engine = MarketMindStorage()
                self.rag_enabled = True
            except Exception as e:
                # If RAG initialization fails, log the error so we aren't left in the dark
                print(f"⚠️ Researcher RAG link failed: {e}")

    def execute_stream(self, message):
        """
        Handles streaming so the user can see the 
        AI thinking in real-time (very useful for Streamlit).
        """
        self.messages.append({"role": "user", "content": message})
        
        # Call GPT-4o-mini (fast and cost-effective for research tasks)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0, # Stay factual, no creativity needed here
            messages=self.messages,
            stream=True  # This allows receiving the response bit by bit
        )
        
        full_response = ""
        for chunk in response:
            # Capture each "piece" of text as it arrives
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content # Return the chunk progressively
        
        # Once finished, store the full response in the history
        self.messages.append({"role": "assistant", "content": full_response})

    def resolve_ticker(self, query):
        """
        This method converts a company name into a stock ticker.
        Example: 'TotalEnergies' becomes 'TTE.PA'.
        """
        # Prepare the specific prompt for stock identification
        prompt = self.ticker_resolver_template.format(query=query)
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0 # We want an ultra-precise response, no hallucinations
        )
        
        # Clean up the response to keep only the code (e.g., 'AAPL')
        ticker = completion.choices[0].message.content.strip()
        return ticker