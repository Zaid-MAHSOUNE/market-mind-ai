import os
import sys
from RAG.rag import MarketMindStorage

def executer_tests():
    print("--- Debut de la verification du systeme RAG ---")

    # 1. Verification de l'initialisation
    try:
        gestionnaire_db = MarketMindStorage()
        print("Confirmation : Connexion a ChromaDB etablie avec succes.")
    except Exception as e:
        print(f"Erreur fatale lors de l'initialisation : {e}")
        return

    # 2. Gestion des documents PDF
    fichier_pdf = "rapport-commision-IA-france-mars-2024.pdf"
    if os.path.exists(fichier_pdf):
        print(f"Analyse du document : {fichier_pdf} en cours...")
        gestionnaire_db.add_document(fichier_pdf)
        print("Statut : Document indexe et segmente.")
    else:
        print("Information : Aucun nouveau fichier PDF detecte. Recherche sur les donnees existantes.")

    # 3. Validation de la recherche semantique
    requete_test = "souverainete numerique"
    print(f"Lancement d'une recherche test sur : '{requete_test}'")
    resultats = gestionnaire_db.search(requete_test)
    
    if resultats:
        print("Succes : Des segments pertinents ont ete extraits de la base de donnees.")
        # Affichage d'un court apercu pour verification visuelle
        apercu = resultats[:150].replace('\n', ' ')
        print(f"Extrait : {apercu}...")
    else:
        print("Attention : La recherche n'a retourne aucun resultat. Verifiez l'indexation.")

    # 4. Test de compatibilite avec l'interface Agent
    print("Verification de la compatibilite avec le framework des agents...")
    outil_rag = gestionnaire_db.get_tool()
    
    if outil_rag.name == "Market_Knowledge_Base":
        reponse_outil = outil_rag.func("Quelles sont les recommandations majeures ?")
        if reponse_outil:
            print("Validation : L'outil est pret a etre integre aux agents de recherche.")
        else:
            print("Echec : L'outil ne parvient pas a extraire de reponse.")
    else:
        print("Erreur : La configuration de l'objet Tool est incorrecte.")

    print("--- Fin des tests du module de stockage ---")

if __name__ == "__main__":
    # Securite : Verification de la presence de la cle OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        print("Erreur : La variable d'environnement OPENAI_API_KEY n'est pas definie.")
    else:
        executer_tests()