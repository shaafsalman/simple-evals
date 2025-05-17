import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from ..model_types import MessageList, SamplerBase, SamplerResponse
from .. import common

class QwenCompletionSampler(SamplerBase):
    """
    Sampler for Qwen3-14B model using HuggingFace transformers.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        enable_thinking: bool = True,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.device_map = device_map
        
        # Set Huggingface cache directory if needed
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        
        print(f"Initializing {model_name} with temperature={temperature}, enable_thinking={enable_thinking}")
        
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            print(f"Tokenizer loaded successfully for {model_name}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=device_map,
                cache_dir=cache_dir
            )
            print(f"Model loaded successfully for {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    
    def _format_messages(self, message_list: MessageList) -> list:
        """Convert message list to the format expected by Qwen."""
        messages = []
        for message in message_list:
            messages.append({"role": message["role"], "content": message["content"]})
        
        return messages
    
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        max_retries = 3
        
        while trial < max_retries:
            try:
                if not common.has_only_user_assistant_messages(message_list):
                    raise ValueError(f"Qwen sampler only supports user and assistant messages, got {message_list}")
                
                # Format messages for Qwen
                messages = self._format_messages(message_list)
                
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking
                )
                
                # Tokenize input
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                # Generate response
                print(f"Generating response with max_new_tokens={self.max_tokens}, temperature={self.temperature}")
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature if self.temperature > 0 else 1e-4,
                    do_sample=(self.temperature > 0),
                )
                
                # Extract only the newly generated tokens
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                
                # Parse thinking content if enabled
                thinking_content = ""
                if self.enable_thinking:
                    try:
                        # Find </think> token (151668) to separate thinking from content
                        index = len(output_ids) - output_ids[::-1].index(151668)
                        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    except ValueError:
                        # If </think> not found, treat everything as content
                        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                else:
                    content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"thinking": thinking_content} if thinking_content else {},
                    actual_queried_message_list=message_list,
                )
                
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(f"Error encountered: {e}. Retrying after {exception_backoff} sec")
                time.sleep(exception_backoff)
                trial += 1
                if trial >= max_retries:
                    raise e  # Re-raise the exception if max retries reached