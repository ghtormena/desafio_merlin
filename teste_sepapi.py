import os
import serpapi
from langchain.tools import tool

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the SerpAPI Wrapper
serp_api_key = os.getenv("SERP_API_KEY")
client = serpapi.Client(api_key=serp_api_key)

# Define a custom tool for search (optional)
@tool
def search_tool(query: str) -> str:
    """Search Google using SerpAPI and return the top results."""
    try:
        result = client.search(
            q=query,
            engine="google",
            location="Austin, Texas",
            hl="en",
            gl="us",
            num=3
        )
        return result
    except Exception as e:
        return {"error": str(e)}

    # Example usage
if __name__ == "__main__":
    query = "Florianópolis"
    results = search_tool(query)
    print(results)
