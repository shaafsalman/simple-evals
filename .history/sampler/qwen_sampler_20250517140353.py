import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..model_types import MessageList, SamplerBase, SamplerResponse
from .. import common

class QwenCompletionSampler(SamplerBase):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        temperature: float = 0.6,
        max_new_tokens: int = 32768,
        top_p: float = 0.95,
        top_k: int = 20,
        enable_thinking: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if not common.has_only_user_assistant_messages(message_list):
            raise ValueError("Qwen sampler only supports user and assistant messages.")

        prompt = self.tokenizer.apply_chat_template(
            message_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
        )[0][inputs.input_ids.shape[-1]:].tolist()

        # Try to split thinking and final output (optional)
        try:
            end_think_token = self.tokenizer.convert_tokens_to_ids("</think>")
            idx = len(output_ids) - output_ids[::-1].index(end_think_token)
        except ValueError:
            idx = 0

        response_text = self.tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()

        return SamplerResponse(
            response_text=response_text,
            response_metadata={},
            actual_queried_message_list=message_list,
        )
