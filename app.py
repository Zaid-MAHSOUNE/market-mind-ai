import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Custom Imports
from agents.agent import InvestigatorAgent
from tools.agent_tools import tools
from tools.rag_storage import get_storage_engine 
from tools.stock_tools import load_prompt

# --- Page Configuration ---
st.set_page_config(
    page_title="MarketMind AI | Chat Investigator",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# --- 1. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System initialized. Knowledge Base connecting... How can I help you today?"}]
if "agent_path" not in st.session_state:
    st.session_state.agent_path = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

def main():
    st.title("🕵️‍♂️ MarketMind: Autonomous Investigation Hub")
    
    # --- PHASE 0: MANDATORY BOOTSTRAP ---
    data_folder = "data"
    chroma_path = os.path.abspath(os.path.join(data_folder, "chroma_db"))
    db_exists = os.path.exists(chroma_path) and any(os.scandir(chroma_path))

    if os.path.exists(data_folder):
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
        if pdf_files and not db_exists and not st.session_state.rag_ready:
            with st.status("🏗️ Knowledge Base not found. Ingesting PDFs...", expanded=True) as status:
                try:
                    storage = get_storage_engine()
                    for file in pdf_files:
                        st.write(f"📄 Processing: **{file}**...")
                        storage.add_document(os.path.join(data_folder, file))
                    st.session_state.rag_ready = True
                    status.update(label="✅ Knowledge Base Built!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Failed to build Knowledge Base: {e}")
        elif db_exists:
            get_storage_engine() 
            st.session_state.rag_ready = True

    # --- Sidebar UI ---
    with st.sidebar:
        st.title("🧪 Step Tracker")
        if st.session_state.agent_path:
            for i, step in enumerate(st.session_state.agent_path):
                st.markdown(f"**Step {i+1}:** {step['tool']}")
        else:
            st.info("No active investigation steps.")
        st.divider()
        if st.session_state.rag_ready:
            st.success("✅ Knowledge Base: Online")
        else:
            st.warning("⚠️ Knowledge Base: Offline")

    # --- Tabs Configuration ---
    tab_chat, tab_path = st.tabs(["💬 Interactive Chat", "Tracks 🛤️ Agent Path"])

    with tab_chat:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with tab_path:
        st.subheader("🛠️ Autonomous Execution Trace")
        if not st.session_state.agent_path:
            st.info("The agent's tool-usage path will appear here.")
        else:
            for i, step in enumerate(st.session_state.agent_path):
                # Distinguish between tool calls and reasoning in the trace
                icon = "⚙️" if "Action" in step['tool'] else "🧠"
                with st.expander(f"{icon} Step {i+1}: {step['tool']}", expanded=False):
                    st.markdown(step['result'])

    # --- Chat Input & Execution ---
    if prompt := st.chat_input("Analyze Financial Strategy..."):
        st.session_state.agent_path = [] 
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.messages[-1]["role"] == "user":
        job_temps = {
        "reasoning": 0,  # Strict precision for tools/logic
        "critique": 0.3    # Tiny bit of creativity for the auditor to find gaps
        }

        llm = ChatOpenAI(model="gpt-4o")
        
        system_instructions = load_prompt("system_instructions.txt")

        agent = InvestigatorAgent(
        llm, 
        tools, 
        system_prompt=system_instructions,
        temps=job_temps
        )

        history = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            # UI Containers for real-time thoughts
            thought_container = st.container() 
            full_response = ""
            
            with st.status("🧠 Investigator is thinking...", expanded=True) as status:
                for output in agent.graph.stream({"messages": history}):
                    for key, value in output.items():
                        
                        if key == "llm":
                            # This is the CoT / Initial Reasoning
                            reasoning = value["messages"][-1].content
                            if reasoning:
                                with thought_container:
                                    with st.expander("💭 Chain of Thought (Reasoning Step)", expanded=False):
                                        st.markdown(reasoning)
                                
                                st.session_state.agent_path.append({
                                    "tool": "LLM Reasoning (CoT)",
                                    "result": reasoning
                                })
                                full_response = reasoning

                        elif key == "action":
                            # This is the ReAct (Action) phase
                            for msg in value["messages"]:
                                st.write(f"🛠️ **ReAct Action:** Using `{msg.name}`")
                                st.session_state.agent_path.append({
                                    "tool": f"Action: {msg.name}",
                                    "result": f"**Input Arguments:** (Generated by AI)\n**Result:**\n{msg.content}"
                                })

                        elif key == "critique":
                            reflexion = value["messages"][-1].content
                            
                            if "---" in reflexion:
                                # 1. Split the text into the Audit and the Plan
                                parts = reflexion.split("---")
                                audit_resume = parts[0].replace("RESUME:", "").strip()
                                revised_plan = parts[1].strip()
                                
                                # 2. Display the Audit Resume as a subtle note
                                st.markdown(f"⚖️ *Audit Findings: {audit_resume}*")
                                
                                # 3. Display the Strategy in the professional Green Box
                                st.success(revised_plan)
                                
                                full_response = revised_plan
                            else:
                                # Fallback: If the model forgets the '---', still use the green box
                                st.success(reflexion)
                                full_response = reflexion

                            # Update the sidebar tracker
                            st.session_state.agent_path.append({
                                "tool": "Reflexion Audit",
                                "result": reflexion
                            })

                status.update(label="✅ Analysis Complete", state="complete", expanded=False)
            
            # Show final strategy prominently
            st.markdown("### 🎯 Final Investment Strategy")
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

if __name__ == "__main__":
    main()