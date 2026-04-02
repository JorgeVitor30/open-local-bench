import ollama
from typing import Any, Type
from pydantic import BaseModel
import json

from src.models.model_abstract import ModelAbstract


class OllamaModel(ModelAbstract):
    """
    Ollama model running locally.
    """
    
    def __init__(self, model_name: str = "llama3.2", host: str = "http://localhost:11434"):
        self._model_name = model_name
        self._host = host
        self._client = ollama.Client(host=host)
    
    @property
    def name(self) -> str:
        return f"ollama/{self._model_name}"
    
    def run(self, prompt: str, **kwargs) -> Any:
        """
        Run prompt on Ollama.
        
        Args:
            prompt: The prompt to send
            response_format: Optional Pydantic model for structured output
            temperature: Sampling temperature (default: 0.7)
            
        Returns:
            String response or Pydantic object if response_format provided
        """
        response_format = kwargs.get("response_format")
        temperature = kwargs.get("temperature", 0.7)
        
        messages = [{"role": "user", "content": prompt}]
        
        if response_format and issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            format_prompt = f"""{prompt}

        Respond ONLY with a JSON object matching this schema:
        {json.dumps(schema, indent=2)}"""
            messages[0]["content"] = format_prompt
        
        response = self._client.chat(
            model=self._model_name,
            messages=messages,
            options={"temperature": temperature}
        )
        
        content = response["message"]["content"]
        
        if response_format and issubclass(response_format, BaseModel):
            try:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    return response_format.model_validate_json(json_str)

                return response_format.model_validate_json(content)
            except Exception as e:
                raise ValueError(f"Failed to parse structured output: {e}\nResponse: {content}")
        
        return content
