import requests
import json

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_TOKEN = "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"  


def query_llm(prompt: str, max_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95) -> str:
    """
    Query the OpenRouter LLM (Gemini 3 4B free) with a text prompt.
    
    Parameters:
        prompt (str): The prompt to send.
        max_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
    
    Returns:
        str: The generated recommendation text.
    """
    payload = {
        "model": "google/gemma-3-4b-it:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        result = response.json()
        
        # Debug: print the raw response to help diagnose its structure
        # print("API response:", result)
        
        # Try to parse the result. Common structures:
        # (a) A dict with a "choices" key (similar to OpenAI's API)
        # (b) A list with a string as first element
        if isinstance(result, dict):
            if "choices" in result:
                # Try accessing a nested message format
                try:
                    return result["choices"][0]["message"]["content"]
                except Exception:
                    return str(result["choices"][0])
            else:
                return str(result)
        elif isinstance(result, list):
            return result[0] if isinstance(result[0], str) else str(result[0])
        else:
            return str(result)
    except Exception as e:
        return f"Exception during API call: {e}"
