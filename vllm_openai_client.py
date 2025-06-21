from typing import List, Dict, Any, Optional
import os
import time
from openai import OpenAI


class VLLMOpenAIClient:
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config["base_url"]
        self.api_key = config.get("api_key", "1")
        self.default_model = config.get("model")
        self.default_max_tokens = config.get("max_tokens", 500)
        self.default_temperature = config.get("temperature", 0.1)
        self.default_stop_tokens = config.get("stop_tokens", None)
        self.default_max_context_length = config.get("max_context_length", 8192)
        self.default_seed = config.get("seed", 42)
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate(self, 
                messages: List[Dict[str, str]], 
                model: Optional[str] = None,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                stop: Optional[List[str]] = None,
                seed: Optional[int] = None) -> str:
        if not all(set(msg.keys()) >= {"role", "content"} for msg in messages):
            raise ValueError("Invalid message format. Each message must have 'role' and 'content' keys.")
        
        model = model or self.default_model
        if not model:
            raise ValueError("Model must be provided either in the constructor or in the generate method")
        
        params = {
            "messages": messages,
            "model": model,
        }
        
        # Add optional parameters if provided or if defaults exist
        if max_tokens is not None or self.default_max_tokens is not None:
            params["max_tokens"] = max_tokens or self.default_max_tokens
        
        if temperature is not None or self.default_temperature is not None:
            params["temperature"] = temperature or self.default_temperature
            
        if stop is not None or self.default_stop_tokens is not None:
            params["stop"] = stop or self.default_stop_tokens
            
        if seed is not None or self.default_seed is not None:
            params["seed"] = seed or self.default_seed
            
        start_time = time.time()
        response = self.client.chat.completions.create(**params)
        end_time = time.time()
        
        completion_tokens = response.usage.completion_tokens
        response_time = end_time - start_time
        tokens_per_second = completion_tokens / response_time if response_time > 0 else 0
        
        result = {
            "answer": response.choices[0].message.content,
            "completion_tokens": completion_tokens,
            "response_time_seconds": response_time,
            "tokens_per_second": tokens_per_second
        }
        return result


if __name__ == "__main__":
    config = {
        "base_url": "http://10.138.16.219:8666/v1",
        "api_key": "1",
        "model": "mistral-24B",
    }
    
    client = VLLMOpenAIClient(config)
    
    response = client.generate(
        messages=[
            {
                "role": "system",
                "content": "Ты добрый и веселый, отвечай на вопросы.",
            },
            {
                "role": "user",
                "content": "Что такое хорошо, а что такое плохо в ozon?",
            }
        ],
    )
    end_time = time.time()
    print(response) 
