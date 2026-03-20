import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Custom Imports
from agents.agent import InvestigatorAgent
from tools.agent_tools import tools
from tools.rag_storage import get_storage_engine # This is critical

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
    
    # --- PHASE 0: MANDATORY BOOTSTRAP (The "Missing" Link) ---
    data_folder = "data"
    chroma_path = os.path.abspath(os.path.join(data_folder, "chroma_db"))
    
    # Check if DB already exists on disk
    db_exists = os.path.exists(chroma_path) and any(os.scandir(chroma_path))

    if os.path.exists(data_folder):
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
        
        # If we have PDFs but NO database, start the ingestion immediately
        if pdf_files and not db_exists and not st.session_state.rag_ready:
            with st.status("🏗️ Knowledge Base not found. Ingesting PDFs...", expanded=True) as status:
                try:
                    # Get the cached storage engine
                    storage = get_storage_engine()
                    for file in pdf_files:
                        st.write(f"📄 Processing: **{file}**...")
                        storage.add_document(os.path.join(data_folder, file))
                    
                    st.session_state.rag_ready = True
                    status.update(label="✅ Knowledge Base Built Successfully!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Failed to build Knowledge Base: {e}")
        
        # If DB exists, just connect to it
        elif db_exists:
            # This call ensures the singleton instance is loaded into cache
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
            st.warning("⚠️ Knowledge Base: Offline (Add PDFs to /data)")

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
                with st.expander(f"Step {i+1}: {step['tool']}", expanded=False):
                    st.code(step['result'], language="markdown")
            
            if st.session_state.messages[-1]["role"] == "assistant":
                st.markdown("### 🎯 Final Conclusion")
                st.success(st.session_state.messages[-1]["content"])

    # --- Chat Input & Execution ---
    if prompt := st.chat_input("Analyze LVMH strategy..."):
        st.session_state.agent_path = [] # Clear path for new turn
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # If the user just asked a question, run the agent
    if st.session_state.messages[-1]["role"] == "user":
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        # Force a professional structure in the persona
        system_instructions = """
        You are a Senior Financial Strategist. 
        You have a memory of this conversation. Use it to provide context.
        When asked for advice, always provide:
        - A Clear Verdict (Buy/Hold/Sell)
        - Entry and Exit prices
        - Risk analysis based on RAG and News.
        Format your final response using professional Markdown.
        """
        agent = InvestigatorAgent(llm, tools, system_prompt=system_instructions)

        # Convert history to LangChain format
        history = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            with st.status("🕵️ Investigating...", expanded=True) as status:
                full_response = ""
                for output in agent.graph.stream({"messages": history}):
                    for key, value in output.items():
                        if key == "llm":
                            # Use content from the very last message in the list
                            full_response = value["messages"][-1].content
                        elif key == "action":
                            for msg in value["messages"]:
                                st.write(f"🛠️ Tool Call: `{msg.name}`")
                                st.session_state.agent_path.append({
                                    "tool": msg.name,
                                    "result": msg.content
                                })
                status.update(label="Analysis Done", state="complete", expanded=False)
            
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

if __name__ == "__main__":
    main()