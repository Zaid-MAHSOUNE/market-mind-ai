import os
from typing import Annotated, TypedDict, List
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from tools.stock_tools import load_prompt

# --- 1. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]

class InvestigatorAgent:
    def __init__(self, model: ChatOpenAI, tools: list, system_prompt: str = "", temps: dict = None):
        self.system = system_prompt
        self.tools = {t.name: t for t in tools}
        
        # Default temperatures if none provided
        self.temps = temps or {"reasoning": 0.3, "critique": 0.0}

        # --- Create specialized models for each node ---
        # 1. Reasoning Model: Logic-focused + Tools
        self.reasoning_model = model.bind_tools(tools).bind(temperature=self.temps["reasoning"])
        
        # 2. Critique Model: Audit-focused + No Tools
        self.critique_model = model

        self.critique_template = load_prompt("critique_prompt.txt")

        # --- Build the Graph ---
        builder = StateGraph(AgentState)

        # Define Nodes
        builder.add_node("llm", self.call_llm)
        builder.add_node("action", self.take_action)
        builder.add_node("critique", self.self_correction)

        # Set Entry Point
        builder.set_entry_point("llm")
        
        # Define Conditional Edges
        builder.add_conditional_edges(
            "llm",
            self.should_continue,
            {
                True: "action", 
                False: "critique"
            }
        )

        # Action leads back to LLM to process results
        builder.add_edge("action", "llm")
        
        # Critique is the final step before the finish line
        builder.add_edge("critique", END)

        self.graph = builder.compile()

    # --- 3. Node & Edge Logic ---

    def should_continue(self, state: AgentState):
        last_message = state['messages'][-1]
        return len(last_message.tool_calls) > 0

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        
        response = self.reasoning_model.invoke(messages) 
        return {"messages": [response]}

    def take_action(self, state: AgentState):
        last_message = state['messages'][-1]
        tool_results = []
        for tool_call in last_message.tool_calls:
            if tool_call['name'] in self.tools:
                result = self.tools[tool_call['name']].invoke(tool_call['args'])
            else:
                result = f"Error: Tool '{tool_call['name']}' not found."
            
            tool_results.append(ToolMessage(tool_call_id=tool_call['id'], name=tool_call['name'], content=str(result)))
        return {"messages": tool_results}

    def self_correction(self, state):
        """Reflexion Node: Audits for over-optimism and suggests a revised strategy."""
        last_analysis = state['messages'][-1].content
        
        reflection_prompt = self.critique_template.format(last_analysis=last_analysis)

        response = self.critique_model.invoke([
            SystemMessage(content=reflection_prompt)
        ])
        
        return {"messages": [response]}