import sys
import os
import time

# Ensure modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.researcher import Researcher
from agents.analyst import Analyst

def print_step(title, content, color="\033[94m"):
    """Helper to print formatted steps for the project demo."""
    reset = "\033[0m"
    print(f"\n{color}{'='*20} {title} {'='*20}{reset}")
    print(content)
    print(f"{color}{'='*50}{reset}")

def test_reasoning_mechanics():
    print("🚀 Starting Detailed Reasoning Mechanics Test...")
    print("Objective: Demonstrate ReAct, Chain of Thought, and Self-Correction.")

    # 1. INITIALIZATION
    ticker = "NVDA"
    researcher = Researcher()
    analyst = Analyst()

    # --- PHASE 1: ReAct (Reason + Act) ---
    print_step("PHASE 1: ReAct (Researcher)", "Agent is planning its search strategy...")
    
    # Simulating the 'Action' part of ReAct
    raw_data = researcher.get_raw_data(ticker)
    print_step("ACT: Data Acquisition", f"Successfully pulled 2026 data for {ticker}.\nSource: Geopolitical News & Yahoo Finance.")

    # Simulating the 'Thought' part of ReAct
    research_summary = researcher(f"Perform a ReAct analysis on this data: {raw_data}")
    print_step("THOUGHT & OBSERVATION", research_summary, "\033[92m") # Green for Thought

    # --- PHASE 2: Chain of Thought (Analyst) ---
    print_step("PHASE 2: Chain of Thought (Analyst)", "Decomposing the researcher's findings into logical steps...")
    
    # We ask the analyst to be explicit about its steps
    cot_prompt = "Using Chain of Thought, break down the risks and opportunities for this ticker."
    analysis_result = analyst(f"{cot_prompt}\n\nDATA: {research_summary}")
    print_step("CHAIN OF THOUGHT STEPS", analysis_result, "\033[93m") # Yellow for CoT

    # --- PHASE 3: Self-Correction (Reflection) ---
    print_step("PHASE 3: Self-Correction (Critic)", "The Analyst will now critique its own logic to ensure high reliability.")
    
    # This forces the "Reflexion" required by your project [cite: 5, 38]
    correction_prompt = "Now, critique your previous analysis. Are there any hallucinations or market biases? Provide a final self-corrected verdict."
    final_verdict = analyst(correction_prompt)
    print_step("FINAL SELF-CORRECTED VERDICT", final_verdict, "\033[95m") # Magenta for Correction

if __name__ == "__main__":
    test_reasoning_mechanics()