import requests
import json

class DeepSeekChat:
    def __init__(self, api_key, model="deepseek-chat", api_base="https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        """Chat with DeepSeek model

        Args:
            messages: Chat history, format like [{"role": "user", "content": "Hello"}]
            temperature: Generation temperature
            max_tokens: Maximum number of tokens to generate

        Returns:
            dict: API response
        """
        url = f"{self.api_base}/chat/completions"
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

# Usage example
if __name__ == "__main__":
    # Replace with your DeepSeek API Key
    API_KEY = "sk-b9b72de3f534424f8bf41f5047f8658b"

    chat = DeepSeekChat(API_KEY)

    # Initialize chat history
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Pool playing techniques"}
    ]

    # Get response
    response = chat.chat(messages)
    print("AI Response:", response["choices"][0]["message"]["content"])

    # Continue conversation
    messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
    messages.append({"role": "user", "content": "How to learn deep learning"})

    response = chat.chat(messages)
    print("AI Response:", response["choices"][0]["message"]["content"])