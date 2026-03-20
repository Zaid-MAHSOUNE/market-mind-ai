import os
from typing import Annotated, TypedDict, List
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

# --- 1. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]

class InvestigatorAgent:
    def __init__(self, model: ChatOpenAI, tools: list, system_prompt: str = ""):
        self.system = system_prompt
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

        # --- 2. Build the Graph ---
        builder = StateGraph(AgentState)

        # Define Nodes
        builder.add_node("llm", self.call_llm)
        builder.add_node("action", self.take_action)
        builder.add_node("critique", self.self_correction) # New Node for Reflexion

        # Set Entry Point
        builder.set_entry_point("llm")
        
        # Define Conditional Edges
        builder.add_conditional_edges(
            "llm",
            self.should_continue,
            {
                True: "action", 
                False: "critique" # Instead of END, go to Self-Correction
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
        response = self.model.invoke(messages)
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

    def self_correction(self, state: AgentState):
        """Reflexion Node: Audits the previous analysis."""
        # Get the analysis from the previous 'llm' node
        last_analysis = state['messages'][-1].content
        
        reflection_prompt = f"""
        ACT AS A SENIOR EDITOR. REVIEW THIS ANALYSIS:
        ---
        {last_analysis}
        ---
        
        CRITICAL INSTRUCTIONS:
        1. Evaluate for risks, pricing accuracy, and RAG data usage.
        2. If errors exist, fix them.
        3. MANDATORY: You must output the COMPLETE FINAL VERSION of the strategy.
        4. NEVER simply say 'No changes needed'. You must REPRINT the full report so the user can see it.
        """
        
        response = self.model.invoke([SystemMessage(content=reflection_prompt)])
        return {"messages": [response]}