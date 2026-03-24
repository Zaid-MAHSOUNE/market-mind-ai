import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Nos imports locaux
from agents.agent import InvestigatorAgent
from tools.agent_tools import tools
from tools.rag_storage import get_storage_engine 
from tools.stock_tools import load_prompt

# --- Configuration de la page ---
st.set_page_config(
    page_title="MarketMind AI | Chat Investigator",
    page_icon="🕵️‍♂️",
    layout="wide" # Mode large pour mieux voir les colonnes et le tracé de l'agent
)

# --- 1. Initialisation des variables de session (Session State) ---
# Essentiel pour que Streamlit ne "perde" pas la mémoire à chaque clic
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Système initialisé. Connexion à la base de connaissances... Comment puis-je vous aider ?"}]
if "agent_path" not in st.session_state:
    st.session_state.agent_path = [] # Pour stocker l'historique des actions de l'IA
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

def main():
    st.title("🕵️‍♂️ MarketMind: Hub d'Investigation Autonome")
    
    # --- PHASE 0 : BOOTSTRAP AUTOMATIQUE ---
    # Ici, on vérifie si on a des PDF à ingérer dès le lancement
    data_folder = "data"
    chroma_path = os.path.abspath(os.path.join(data_folder, "chroma_db"))
    db_exists = os.path.exists(chroma_path) and any(os.scandir(chroma_path))

    if os.path.exists(data_folder):
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
        # Si on a des PDF mais pas de base de données, on lance l'ingestion automatique
        if pdf_files and not db_exists and not st.session_state.rag_ready:
            with st.status("🏗️ Base de connaissances introuvable. Indexation des PDF...", expanded=True) as status:
                try:
                    storage = get_storage_engine()
                    for file in pdf_files:
                        st.write(f"📄 Traitement de : **{file}**...")
                        storage.add_document(os.path.join(data_folder, file))
                    st.session_state.rag_ready = True
                    status.update(label="✅ Base de connaissances prête !", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Échec de la création de la base : {e}")
        elif db_exists:
            get_storage_engine() 
            st.session_state.rag_ready = True

    # --- Barre latérale (Sidebar) ---
    with st.sidebar:
        st.title("🧪 Suivi des étapes")
        # Affiche en temps réel ce que l'agent est en train de faire (quel outil il utilise)
        if st.session_state.agent_path:
            for i, step in enumerate(st.session_state.agent_path):
                st.markdown(f"**Étape {i+1}:** {step['tool']}")
        else:
            st.info("Aucune investigation en cours.")
        st.divider()
        # Indicateur visuel pour le statut du RAG
        if st.session_state.rag_ready:
            st.success("✅ Base de connaissances : En ligne")
        else:
            st.warning("⚠️ Base de connaissances : Hors ligne")

    # --- Organisation en Onglets ---
    tab_chat, tab_path = st.tabs(["💬 Chat Interactif", "🛤️ Tracé de l'Agent"])

    with tab_chat:
        # Affichage classique des messages du chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with tab_path:
        # C'est ici qu'on inspecte les "entrailles" du raisonnement de l'IA
        st.subheader("🛠️ Trace d'exécution autonome")
        if not st.session_state.agent_path:
            st.info("Le chemin parcouru par l'agent s'affichera ici.")
        else:
            for i, step in enumerate(st.session_state.agent_path):
                icon = "⚙️" if "Action" in step['tool'] else "🧠"
                with st.expander(f"{icon} Étape {i+1}: {step['tool']}", expanded=False):
                    st.markdown(step['result'])

    # --- Saisie utilisateur et exécution ---
    if prompt := st.chat_input("Analysez une stratégie financière..."):
        st.session_state.agent_path = [] # On reset le tracé pour la nouvelle question
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # On force le rafraîchissement pour afficher le message user immédiatement

    # Logique de réponse de l'agent
    if st.session_state.messages[-1]["role"] == "user":
        # On définit les températures : 0 pour être précis, 0.3 pour l'audit final
        job_temps = {"reasoning": 0, "critique": 0.3}

        llm = ChatOpenAI(model="gpt-4o")
        system_instructions = load_prompt("system_instructions.txt")

        # Initialisation de notre agent d'investigation
        agent = InvestigatorAgent(
            llm, 
            tools, 
            system_prompt=system_instructions,
            temps=job_temps
        )

        # Conversion du format Streamlit vers le format LangChain (Messages)
        history = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            thought_container = st.container() 
            full_response = ""
            
            # On utilise .stream() pour voir les étapes du graphe en direct
            with st.status("🧠 L'enquêteur réfléchit...", expanded=True) as status:
                for output in agent.graph.stream({"messages": history}):
                    for key, value in output.items():
                        
                        if key == "llm":
                            # Phase de réflexion (Chain of Thought)
                            reasoning = value["messages"][-1].content
                            if reasoning:
                                with thought_container:
                                    with st.expander("💭 Raisonnement interne", expanded=False):
                                        st.markdown(reasoning)
                                
                                st.session_state.agent_path.append({
                                    "tool": "Raisonnement LLM (CoT)",
                                    "result": reasoning
                                })
                                full_response = reasoning

                        elif key == "action":
                            # Phase ReAct : L'IA utilise un outil
                            for msg in value["messages"]:
                                st.write(f"🛠️ **Action :** Utilisation de `{msg.name}`")
                                st.session_state.agent_path.append({
                                    "tool": f"Action: {msg.name}",
                                    "result": f"**Arguments :** (Générés par l'IA)\n**Résultat :**\n{msg.content}"
                                })

                        elif key == "critique":
                            reflexion = value["messages"][-1].content
                            
                            if "---" in reflexion:
                                parts = reflexion.split("---")
                                audit_resume = parts[0].replace("RESUME:", "").strip()
                                revised_plan = parts[1].strip()
                                
                                import re
                                price_match = re.search(r"Target Entry Price:\*\* ([\$\d\.,\-\s]+)", revised_plan)
                                target_price = price_match.group(1).strip() if price_match else "N/A"
                                
                                # Display Audit Findings
                                st.markdown(f"⚖️ *Audit Findings: {audit_resume}*")
                                
                                # Create a clean layout with a Metric for the Price
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.metric(label="Target Entry Price", value=target_price)
                                    
                                with col2:
                                    # Color-coded boxes based on Verdict
                                    if "Verdict: Buy" in revised_plan:
                                        st.success(revised_plan)
                                    elif "Verdict: Sell" in revised_plan:
                                        st.error(revised_plan)
                                    else:
                                        st.warning(revised_plan)
                                    
                                full_response = revised_plan
                            else:
                                st.info(reflexion)
                                full_response = reflexion

                            st.session_state.agent_path.append({"tool": "Price Discovery Audit", "result": reflexion})

                            st.session_state.agent_path.append({
                                "tool": "Strategist Audit",
                                "result": reflexion
                            })

                status.update(label="✅ Analyse terminée", state="complete", expanded=False)
            
            # Affichage final propre sous le status
            st.markdown("### 🎯 Stratégie d'Investissement Finale")
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

if __name__ == "__main__":
    main()