import os
from typing import Annotated, TypedDict, List
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from tools.stock_tools import load_prompt

# --- 1. State Definition ---
# The TypedDict defines the structure of the data flowing through the graph.
class AgentState(TypedDict):
    # Annotated with 'add' allows messages to be appended to the list 
    # instead of overwriting it at each step.
    messages: Annotated[List[BaseMessage], add]

class InvestigatorAgent:
    def __init__(self, model: ChatOpenAI, tools: list, system_prompt: str = "", temps: dict = None):
        self.system = system_prompt
        # Creating a mapping dictionary to call tools by their name
        self.tools = {t.name: t for t in tools}
        
        # Default temperatures: 0.3 for logic/reasoning, 0.0 for a strict critique
        self.temps = temps or {"reasoning": 0.3, "critique": 0.0}

        # --- Creating specialized models for each node ---
        
        # 1. Reasoning Model: Allowed to use tools + flexible temperature
        self.reasoning_model = model.bind_tools(tools).bind(temperature=self.temps["reasoning"])
        
        # 2. Critique Model: Pure audit, no tools allowed to prevent infinite loops
        self.critique_model = model

        # Loading the prompt template for the self-correction phase
        self.critique_template = load_prompt("critique_prompt.txt")

        # --- Building the Graph ---
        builder = StateGraph(AgentState)

        # Node Definitions (The computational steps)
        builder.add_node("llm", self.call_llm)           # Analysis and decision making
        builder.add_node("action", self.take_action)     # Tool execution
        builder.add_node("critique", self.self_correction) # Final audit

        # Graph Entry Point
        builder.set_entry_point("llm")
        
        # Defining Conditional Edges
        # After the 'llm' node, we check if we should continue or finish
        builder.add_conditional_edges(
            "llm",
            self.should_continue,
            {
                True: "action",    # If AI wants to use a tool -> head to 'action'
                False: "critique"  # If AI is done thinking -> head to 'critique'
            }
        )

        # Action always leads back to the LLM to analyze the tool's result
        builder.add_edge("action", "llm")
        
        # Critique is the final step before the finish line (END)
        builder.add_edge("critique", END)

        # Compiling the graph to make it operational
        self.graph = builder.compile()

    # --- 3. Node & Edge Logic ---

    def should_continue(self, state: AgentState):
        """Determines if the LLM has requested the use of tools."""
        last_message = state['messages'][-1]
        # If the last message contains 'tool_calls', we proceed to 'action'
        return len(last_message.tool_calls) > 0

    def call_llm(self, state: AgentState):
        """Reasoning Node: Calls the LLM with the current context."""
        messages = state['messages']
        # Prepends the system prompt if one is defined
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        
        response = self.reasoning_model.invoke(messages) 
        # Returns the produced message to be added to the state
        return {"messages": [response]}

    def take_action(self, state: AgentState):
        """Action Node: Actually executes the tools requested by the AI."""
        last_message = state['messages'][-1]
        tool_results = []
        
        # Iterating through every tool call detected in the last message
        for tool_call in last_message.tool_calls:
            if tool_call['name'] in self.tools:
                # Executing the tool with the arguments provided by the AI
                result = self.tools[tool_call['name']].invoke(tool_call['args'])
            else:
                result = f"Error: Tool '{tool_call['name']}' does not exist."
            
            # Creating the tool return message (required for LangChain)
            tool_results.append(ToolMessage(
                tool_call_id=tool_call['id'], 
                name=tool_call['name'], 
                content=str(result)
            ))
        return {"messages": tool_results}

    def self_correction(self, state):
        """Reflection Node: Audits the analysis to correct for over-optimism."""
        # Retrieving the content of the last analysis produced
        last_analysis = state['messages'][-1].content
        
        # Preparing the reflection prompt with the analysis text
        reflection_prompt = self.critique_template.format(last_analysis=last_analysis)

        # Calling the critique model (usually with 0.0 temperature)
        response = self.critique_model.invoke([
            SystemMessage(content=reflection_prompt)
        ])
        
        return {"messages": [response]}