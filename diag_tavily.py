
import toml
import os
from tavily import TavilyClient

print("---" + " Tavily API Connectivity Test" + "---")

try:
    # Find and read the secrets.toml file
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        print(f"Error: Could not find the secrets file at {secrets_path}")
        print("Please make sure the file exists and you have run the script from the project root directory.")
    else:
        with open(secrets_path, "r", encoding="utf-8") as f:
            secrets = toml.load(f)
        
        api_key = secrets.get("TAVILY_API_KEY")

        if not api_key or not api_key.startswith("tvly-"):
            print("Error: TAVILY_API_KEY not found or invalid in .streamlit/secrets.toml")
            print("Please ensure your key starts with 'tvly-'.")
        else:
            print("API Key found. Initializing Tavily client...")
            client = TavilyClient(api_key=api_key)
            
            test_query = "Ehime Prefecture"
            print(f"Performing a basic search with query: '{test_query}'")
            
            try:
                # Using a 30-second timeout for the test
                response = client.search(query=test_query, search_depth="basic", timeout=30)
                print("\n---" + " SUCCESS!" + "---")
                print("Successfully received a response from Tavily API.")
                # Print only the answer if it exists, otherwise the whole response
                if isinstance(response, dict) and "answer" in response:
                    print("\nAnswer: " + response["answer"])
                else:
                    print("\nResponse: " + str(response))

            except Exception as e:
                print(f"\n---" + " FAILURE" + "---")
                print(f"The test failed. An error occurred while trying to connect to Tavily API: {e}")
                print("\nThis suggests a potential network issue, firewall block, or an invalid API key.")

except Exception as e:
    print(f"\nAn unexpected error occurred while setting up the test: {e}")

print("\n---" + " Test Finished" + "---")
