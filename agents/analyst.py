import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Analyst:
    def __init__(self):
        # Path to the analyst logic (CoT + Self-Correction) [cite: 28, 36]
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'analyst_logic.txt')
        with open(prompt_path, 'r') as f:
            self.system = f.read()
        
        self.messages = [{"role": "system", "content": self.system}]

    def __call__(self, research_results):
        # The user message for the analyst is the data from the researcher
        prompt = f"Please analyze this research and perform a self-correction: {research_results}"
        self.messages.append({"role": "user", "content": prompt})
        
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2, # Slightly higher to allow for critical evaluation [cite: 5]
            messages=self.messages
        )
        return completion.choices[0].message.content