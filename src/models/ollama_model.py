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
            
            fields_desc = []
            example_fields = {}
            for field_name, field_info in schema.get("properties", {}).items():
                field_type = field_info.get("type", "string")
                field_desc = field_info.get("description", f"{field_name} value")
                fields_desc.append(f"- {field_name}: {field_type} - {field_desc}")
                if field_type == "string":
                    example_fields[field_name] = f"example_{field_name}"
                elif field_type == "number":
                    example_fields[field_name] = 42
                elif field_type == "boolean":
                    example_fields[field_name] = True
                elif field_type == "array":
                    example_fields[field_name] = []
                elif field_type == "object":
                    example_fields[field_name] = {}
                else:
                    example_fields[field_name] = f"{field_name}_value"
            
            fields_text = "\n".join(fields_desc)
            example_json = json.dumps(example_fields)
            
            format_prompt = f"""{prompt}

            IMPORTANT: You must respond with a valid JSON object containing the actual values, NOT the schema definition.

            The JSON object must have these exact fields:
            {fields_text}

            Example response format:
            {example_json}

            Respond ONLY with the JSON object, no markdown code blocks."""
            messages[0]["content"] = format_prompt
        
        response = self._client.chat(
            model=self._model_name,
            messages=messages,
            options={"temperature": temperature},
            format="json" if response_format else None
        )
        
        content = response["message"]["content"]
        
        if response_format and issubclass(response_format, BaseModel):
            try:
                return response_format.model_validate_json(content)
            except Exception:
                try:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        return response_format.model_validate_json(json_str)
                    raise ValueError("No JSON found in response")
                except Exception as e:
                    raise ValueError(f"Failed to parse structured output: {e}\nResponse: {content}")
        
        return content
