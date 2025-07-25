import os
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("💥 Error: OpenAI API key not found. Please set it in the .env file.")
        return

    print("✅ Environment setup is correct. API key loaded.")
    
    # We will add the logic to run the ingestion and querying here later...

if __name__ == "__main__":
    main()