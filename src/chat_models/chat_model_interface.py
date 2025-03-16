import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
from threading import Thread


class ChatModelInterface:
    def __init__(
        self,
        model_id: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        offload_folder: str = "offload",
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
    ):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            offload_folder=offload_folder,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_id)
        self.model.generation_config.pad_token_id = (
            self.model.generation_config.eos_token_id
        )
        self.device = self.model.device

    @torch.inference_mode()
    def invoke(
        self, message: str, max_new_tokens: int = 3000, temperature: float = 0.1
    ) -> str:
        try:
            input_ids = self.tokenizer(message, return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens, temperature=temperature
            )

            result = self.tokenizer.decode(
                outputs[0][input_ids.shape[1] :], skip_special_tokens=True
            )

            # Manually free tensor memory
            self.close()

            return result
        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            return ""

    def stream(
        self, message: str, max_new_tokens: int = 3000, temperature: float = 0.1
    ):
        try:
            input_ids = self.tokenizer(
                message, return_tensors="pt", padding=True
            ).input_ids.to(self.device)
            streamer = TextIteratorStreamer(self.tokenizer)

            generation_kwargs = {
                "input_ids": input_ids,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                yield new_text

            self.close()
        except Exception as e:
            print(f"[Error] Streaming failed: {e}")

    def close(self):
        """
        Clears unnecessary GPU memory without unloading the model.
        """
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("[Info] Cleared unused GPU memory.")
        except Exception as e:
            print(f"[Error] Failed to clear GPU memory: {e}")
