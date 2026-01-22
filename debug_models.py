import google.generativeai as genai
import os

def list_available_models():
    """
    Lists all models available to the API key that support content generation.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("\n\033[91mError: GOOGLE_API_KEY environment variable is not set.\033[0m")
        print("Please set it by running: export GOOGLE_API_KEY='your_key_here'")
        print("Or run the script with the key inline: GOOGLE_API_KEY='...' python debug_models.py")
        return

    try:
        genai.configure(api_key=api_key)
        
        print(f"\nScanning available models for key: {api_key[:5]}...*************\n")
        
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
                print(f"- \033[92m{m.name}\033[0m (Display Name: {m.display_name})")
        
        if not available_models:
            print("\nNo models found that support 'generateContent'. Check your API key permissions and region.")
        else:
            print(f"\n\033[94mFound {len(available_models)} models compatible with Chat/Generation.\033[0m")
            
    except Exception as e:
        print(f"\n\033[91mAn error occurred while listing models:\033[0m {str(e)}")

if __name__ == "__main__":
    list_available_models()
