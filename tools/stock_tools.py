import yfinance as yf
from tavily import TavilyClient
import os
from dotenv import load_dotenv

# On charge les clés API (Tavily, etc.) depuis le fichier .env
load_dotenv()

# Initialisation de Tavily pour les recherches web (le moteur de recherche spécialisé IA)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def get_stock_prices(ticker: str, period: str = "1mo"):
    """
    Va chercher l'historique des prix pour un ticker (ex: 'AAPL' ou 'TTE.PA').
    On récupère les 10 derniers jours pour avoir une idée de la tendance.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    # On ne garde que la clôture et le volume, c'est plus lisible pour l'IA
    return df[['Close', 'Volume']].tail(10).to_string()

def get_market_news(query: str):
    """
    C'est l'outil 'enquêteur'. Il fouille le web en profondeur via Tavily 
    pour dénicher les news fraîches et les tendances de marché.
    """
    # On utilise le mode 'advanced' pour avoir des résultats de meilleure qualité
    search_result = tavily.search(query=query, search_depth="advanced", max_results=5)
    return search_result

def get_company_info(ticker):
    """
    Récupère le profil d'une boîte. J'ai ajouté des sécurités pour éviter 
    que le script ne plante si la boîte est privée ou peu connue.
    """
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # On essaie de choper le nom et le secteur, sinon on met des valeurs par défaut
        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "Entité Privée/Gouvernementale")
        
        # --- FIX DU CRASH --- 
        # On vérifie si le résumé existe avant de tenter de le couper (slicing),
        # sinon ça lève une erreur sur les boîtes non cotées.
        summary = info.get("longBusinessSummary")
        if summary:
            summary = summary[:500] + "..." # On tronque pour ne pas saturer le contexte de l'IA
        else:
            summary = "Pas de résumé public disponible (Boîte peut-être privée ou nationalisée)."
            
        return {
            "name": name,
            "sector": sector,
            "summary": summary
        }
    except Exception as e:
        # Si yfinance fait des siennes, on renvoie au moins le ticker et l'erreur
        return {"name": ticker, "sector": "Inconnu", "summary": f"Erreur lors de la récupération : {str(e)}"}
    
def load_prompt(filename):
    path = os.path.join("prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    