import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.researcher import Researcher
from agents.analyst import Analyst

def run_test():
    print("🚀 Starting Manual Agent Test (ReAct + CoT + Self-Correction)")
    
    ticker = "NVDA"
    researcher = Researcher()
    analyst = Analyst()

    # 1. Action: Get real-time data [cite: 37]
    print(f"--- Step 1: Researcher Gathering Data for {ticker} ---")
    data_context = researcher.get_raw_data(ticker)
    
    # 2. Reasoning: Researcher processes the data [cite: 33]
    print("--- Step 2: Researcher Reasoning Loop ---")
    research_summary = researcher(f"Analyze this data and find the geopolitical links: {data_context}")
    print(f"Researcher Result: {research_summary[:150]}...\n")

    # 3. Self-Correction: Analyst critiques the findings 
    print("--- Step 3: Analyst Performing Self-Correction & Final Verdict ---")
    final_verdict = analyst(research_summary)
    
    print("\n✅ FINAL ANALYST OUTPUT:")
    print(final_verdict)

if __name__ == "__main__":
    run_test()