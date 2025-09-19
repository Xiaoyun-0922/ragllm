import requests
import json
from typing import Dict, List
from my_config import config

class DeepSeekAPI:
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.api_url = config.DEEPSEEK_API_URL

    def generate(self, prompt: str, **kwargs) -> str:
        """Call DeepSeek API to generate response"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek-chat",  # DeepSeek model name
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # More stable and faster
            "max_tokens": 512,   # Reduce generation length to lower latency
            "top_p": 0.9,
            **kwargs  # Other optional parameters
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()  # Check HTTP errors
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"DeepSeek API call failed: {e}")
            return "Sorry, an error occurred while generating the response."