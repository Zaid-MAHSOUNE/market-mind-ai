import os
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

# Import your tools logic
from tools.council_tools import (
    get_analyst_research, 
    get_sentiment_data, 
    get_risk_metrics, 
    get_stock_technical_data
)

load_dotenv()

# --- 1. DATA STRUCTURE ---
class FinancialMission(BaseModel):
    """Extraction model for the Orchestrator"""
    symbol: str = Field(..., description="The stock ticker symbol.")
    analysis_duration: str = Field("1y", description="Time window for historical analysis.")
    investment_horizon: str = Field("3y", description="Intended hold period.")
    budget: str = Field("$10,000", description="User's investment budget.")

# --- 2. AGENT DEFINITIONS ---

orchestrator_agent = Agent(
    role='Mission Controller',
    goal='Accurately extract the target ticker and investor constraints from: {user_prompt}',
    backstory="""You are an expert at parsing messy human language into clean financial 
    parameters. You identify the stock symbol and the user's budget, ensuring the 
    specialists have a clear target to hit.""",
    verbose=True,
    allow_delegation=True
)

analyst_agent = Agent(
    role='Deep-Value Detective',
    goal='Examine the company’s "engine room"—earnings, cash flow, and competitive moat.',
    backstory="""You hate hype. You only care about the numbers. You look for 
    profitability, revenue growth, and whether the company actually makes more 
    money than it spends. You provide the 'Hard Truth' about the stock's value.""",
    tools=[get_analyst_research, get_stock_technical_data],
    verbose=True
)

sentiment_agent = Agent(
    role='Market Pulse Decoder',
    goal='Identify if the market is currently driven by "Fear" or "Greed" regarding this stock.',
    backstory="""You are a behavioral finance expert. You scan social media, 
    news trends, and retail investor chatter to see if a stock is over-hyped 
    or unfairly hated. You provide the 'Vibe Check' of the market.""",
    tools=[get_sentiment_data],
    verbose=True
)

risk_agent = Agent(
    role='Safety First Specialist',
    goal='Identify exactly how much money the user could lose and how to prevent it.',
    backstory="""You are a capital preservation expert. Your job is to find the 
    hidden traps. You calculate volatility and tell the user when a 'great' 
    investment is actually a 'dangerous' gamble based on their budget.""",
    tools=[get_risk_metrics],
    verbose=True
)

visualizer_agent = Agent(
    role='Plain-English Portfolio Advisor',
    goal='Summarize the council’s complex findings into a simple, jargon-free investment story.',
    backstory="""You are a world-class communicator. You take the complex data from 
    the Detective, the Decoder, and the Safety Specialist and turn it into a 
    story a 5th grader could understand. You focus on 'What this means for your money'.""",
    tools=[get_stock_technical_data], 
    verbose=True
)

# --- 3. TASK DEFINITIONS ---
orchestration_task = Task(
    description="Analyze '{user_prompt}' and extract symbol, duration, horizon, and budget.",
    expected_output="A structured mission summary with the target ticker.",
    agent=orchestrator_agent,
    output_json=FinancialMission
)

analysis_task = Task(
    description="Perform fundamental analysis based on the mission defined by the strategist.",
    expected_output="Detailed report on financial health and growth prospects.",
    agent=analyst_agent,
    context=[orchestration_task]
)

sentiment_task = Task(
    description="Analyze current market sentiment for the ticker identified in the mission.",
    expected_output="A sentiment summary and key psychological drivers.",
    agent=sentiment_agent,
    context=[orchestration_task]
)

risk_task = Task(
    description="Calculate risk metrics based on the identified ticker and budget.",
    expected_output="A risk profile including VaR analysis.",
    agent=risk_agent,
    context=[orchestration_task]
)

visual_task = Task(
    description="""Review all Council findings and write a 'Friendly Investor Brief'.
    
    1. **The Big Picture (The Why)**: In 3 sentences, explain why this stock is or isn't 
       a good fit. Use an analogy (e.g., 'This stock is like a reliable old truck...') 
       to make it easy to understand.
       
    2. **The Traffic Light View**: 
       - 🟢 **The Good**: What is the #1 reason to buy? (Simple English).
       - 🟡 **The Watch-Out**: What is the #1 thing to be careful of? (Simple English).
       - 🔴 **The Risk**: If things go wrong, how much could you lose?

    3. **The Bottom Line**: Give a clear 'Yes', 'No', or 'Wait for a lower price' verdict.

    """,
    expected_output="A simple, friendly investment brief followed by the JSON data block.",
    agent=visualizer_agent,
    context=[analysis_task, sentiment_task, risk_task]
)

# --- 4. CREW ---
investment_council = Crew(
    agents=[orchestrator_agent, analyst_agent, sentiment_agent, risk_agent, visualizer_agent],
    tasks=[orchestration_task, analysis_task, sentiment_task, risk_task, visual_task],
    process=Process.sequential,
    verbose=True,
    stream=True
)