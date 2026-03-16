# 📈 MarketMind AI: Intelligent Financial Investigator

**MarketMind AI** is an advanced generative AI application built with **Streamlit** that acts as an autonomous financial analyst. Unlike a standard chatbot, this tool uses a **Multi-Agent architecture** and **ReAct reasoning loops** to search for market trends, analyze stock data, and provide time-sensitive investment recommendations.

## 🎯 Project Objective

The goal is to move beyond simple text generation by implementing an agent capable of **planning, critiquing, and deciding**. The agent doesn't just give a stock name; it justifies the "why" (reasoning), the "when" (timing), and the "how long" (duration) based on real-time data.

## 🧠 Reasoning Techniques Implemented

To ensure high reliability and logical consistency, the following techniques are utilized:

*  **ReAct (Reason + Act):** The agent follows a continuous loop of **Thought → Action → Observation → Response**. It searches for news, observes the market sentiment, and adjusts its logic before providing a final output.


*  **Chain of Thought (CoT):** The model is forced to decompose complex financial indicators into step-by-step logical segments (e.g., analyzing macro trends before micro-level stock data).


*  **Self-Correction (Reflexion):** A "Critic" agent reviews the initial investment hypothesis to detect potential hallucinations or logical gaps before the user sees the recommendation.



## 🛠️ Key Features

*  **Real-time Trend Discovery:** Automatically scrapes and identifies trending sectors in the current stock market.


*  **Deep Financial Analysis:** Analyzes news sentiment and technical data to justify recommendations.


*  **Investment Strategy:** Provides specific entry points (timing) and exit strategies (duration).
*  **Interactive UI:** A clean Streamlit interface that displays the agent's internal "thought process".



## 🚀 Getting Started

1. **Clone the repo:** `git clone https://github.com/Zaid-MAHSOUNE/market-mind-ai.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run the app:** `streamlit run app.py`
