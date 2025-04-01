from google import genai
from google.genai import types
import json
from .prompts import SYSTEM_PROMPT, build_full_prompt

# Initialize the Gemini API client
API_KEY = "AIzaSyDfvojCflHjso_MX67YVZaBULVYSlLv84A"
client = genai.Client(api_key=API_KEY)

def query_llm(prompt: str, max_tokens: int = 1024) -> str:
    """
    Query the Gemini model with a text prompt.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=max_tokens,  # Using increased token limit
                temperature=0.7  # Added temperature for consistent responses
            ),
            contents=prompt
        )
        
        if response.text:
            return response.text
        return "Error: Empty response from model"
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"Error generating response: {str(e)}"

def get_recommendations(cluster_profile: dict) -> str:
    """
    Get marketing recommendations for a cluster based on its profile.
    """
    try:
        # Extract customer information
        cluster = cluster_profile.get('cluster', 'N/A')
        gender = cluster_profile.get('gender', 'N/A')
        campaign_channel = cluster_profile.get('campaign_channel', 'N/A')
        campaign_type = cluster_profile.get('campaign_type', 'N/A')
        
        # Build detailed prompt
        detailed_prompt = (
            f"Customer Profile:\n"
            f"- Cluster: {cluster}\n"
            f"- Gender: {gender}\n"
            f"- Previous Campaign Channel: {campaign_channel}\n"
            f"- Previous Campaign Type: {campaign_type}\n\n"
            "Based on this customer's profile and your knowledge of cluster characteristics, "
            "provide marketing recommendations following this structure:\n"
            "1. Segment Overview (including size and typical characteristics)\n"
            "2. Value Assessment\n"
            "3. Engagement Strategy\n"
            "4. Campaign Recommendations\n"
            "5. Success Metrics\n\n"
            "Format the response using clear sections and bullet points."
        )
        
        return query_llm(detailed_prompt)
        
    except Exception as e:
        print(f"Error building recommendation: {str(e)}")
        return f"Error generating recommendations: {str(e)}"