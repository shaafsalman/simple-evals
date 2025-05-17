import os
import time
import json
import requests
from typing import Any, Optional, List, Dict

from ..model_types import MessageList, SamplerBase, SamplerResponse


class QwenCompletionSampler(SamplerBase):
    """
    Sample from Qwen models using vLLM server API endpoint
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        api_url: str = "http://localhost:8000/v1/completions",
        api_key: str = "test",
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _pack_message(self, content: str, role: str) -> Dict[str, str]:
        """Pack a message with role and content."""
        return {"role": role, "content": content}

    def _handle_text(self, text: str) -> Dict[str, Any]:
        """Handle text content."""
        return {"type": "text", "text": text}

    def _prepare_messages(self, message_list: MessageList) -> MessageList:
        """Prepare messages, adding system message if provided."""
        prepared_messages = message_list.copy()
        if self.system_message and not any(msg.get("role") == "system" for msg in prepared_messages):
            prepared_messages.insert(0, {"role": "system", "content": self.system_message})
        return prepared_messages

    def _format_prompt(self, messages: MessageList) -> str:
        """Format messages into a prompt string for the vLLM API."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
        # Add the final assistant prompt to signal where the model should respond
        prompt += "<|im_start|>assistant\n"
        return prompt

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Generate a response using the Qwen model via vLLM API.
        
        Args:
            message_list: List of message dictionaries with role and content
            
        Returns:
            SamplerResponse object with response text and metadata
        """
        # Prepare messages
        prepared_messages = self._prepare_messages(message_list)
        
        # Format the prompt for vLLM API
        prompt = self._format_prompt(prepared_messages)
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Add reasoning mode parameters if enabled
        if self.enable_thinking:
            payload["enable_reasoning"] = True
        
        trial = 0
        max_retries = 3
        
        while trial < max_retries:
            try:
                # Send the request to the vLLM API
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=180  # 3-minute timeout
                )
                
                # Check for successful response
                response.raise_for_status()
                
                # Parse the JSON response
                result = response.json()
                
                # Extract the generated text
                content = result.get("choices", [{}])[0].get("text", "").strip()
                
                # Extract usage information if available
                usage = result.get("usage", {})
                if not usage:
                    # Create basic usage info if not provided
                    usage = {
                        "prompt_tokens": len(prompt) // 4,  # Rough estimate
                        "completion_tokens": len(content) // 4,  # Rough estimate
                        "total_tokens": (len(prompt) + len(content)) // 4
                    }
                
                # Create metadata
                metadata = {
                    "usage": usage,
                    "model": self.model_name,
                }
                
                # Return the sampler response with the raw text
                return SamplerResponse(
                    response_text=content,
                    response_metadata=metadata,
                    actual_queried_message_list=prepared_messages,
                )
                
            except requests.RequestException as e:
                exception_backoff = 2**trial  # exponential back off
                time.sleep(exception_backoff)
                trial += 1
                
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                time.sleep(exception_backoff)
                trial += 1
        
        # If all retries failed
        return SamplerResponse(
            response_text="Error: Failed to generate response after multiple attempts.",
            response_metadata={"error": "API communication failed"},
            actual_queried_message_list=prepared_messages,
        )g