import mlx.core as mx
from typing import Optional, List, Dict, Any, Iterator
from .model import GPTOSSModel
from .config import GPTOSSConfig
from ..harmony_template import HarmonyTemplateRenderer, create_harmony_template


class TokenGenerator:
    """Token generation utilities for GPT-OSS MLX models."""
    
    def __init__(self, checkpoint: str, use_harmony: bool = True):
        self.model = GPTOSSModel.from_pretrained(checkpoint)
        self.use_harmony = use_harmony
        self.harmony_renderer = HarmonyTemplateRenderer() if use_harmony else None
        # In production, this would load the actual tokenizer
        # For now, use a dummy that matches the interface
        from gpt_oss.tokenizer import get_tokenizer
        self.tokenizer = get_tokenizer()
    
    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
        return_logits: bool = False,
        system_message: Optional[str] = None
    ):
        """Generate tokens autoregressively with optional harmony formatting."""
        # If using harmony format, ensure proper formatting
        if self.use_harmony and self.harmony_renderer:
            # Convert tokens back to text for harmony processing
            try:
                prompt_text = self.tokenizer.decode(prompt_tokens)
                harmony_formatted = create_harmony_template(
                    prompt=prompt_text,
                    system_message=system_message
                )
                prompt_tokens = self.tokenizer.encode(harmony_formatted)
            except Exception as e:
                print(f"Warning: Harmony formatting failed: {e}. Using original tokens.")
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        cache = None
        
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            # Convert tokens to MLX array
            input_ids = mx.array(tokens).reshape(1, -1)
            
            # Forward pass
            if cache is None:
                logits, cache = self.model(input_ids)
                next_logits = logits[0, -1, :]  # Get last token logits
            else:
                # Use only last token with cache
                last_token = mx.array([tokens[-1]]).reshape(1, 1)
                logits, cache = self.model(last_token, cache=cache)
                next_logits = logits[0, 0, :]
            
            # Apply temperature and sample
            if temperature == 0.0:
                predicted_token = int(mx.argmax(next_logits).item())
            else:
                next_logits = next_logits / temperature
                probs = mx.softmax(next_logits, axis=-1)
                predicted_token = int(mx.random.categorical(mx.log(probs)).item())
            
            tokens.append(predicted_token)
            num_generated_tokens += 1
            
            # Yield token and optionally logprobs/logits
            if return_logprobs or return_logits:
                result = {'token': predicted_token}
                
                if return_logprobs:
                    logprobs = mx.log_softmax(next_logits, axis=-1)
                    selected_logprobs = float(logprobs[predicted_token].item())
                    result['logprob'] = selected_logprobs
                    result['all_logprobs'] = logprobs
                
                if return_logits:
                    result['logits'] = next_logits
                    result['all_logits'] = next_logits
                
                yield result
            else:
                yield predicted_token
            
            # Check for stop tokens (including harmony stop tokens)
            harmony_stop_tokens = [200002, 200012]  # <|return|> and <|call|>
            if predicted_token in stop_tokens or (self.use_harmony and predicted_token in harmony_stop_tokens):
                break
    
    def generate_with_harmony(
        self,
        prompt: str,
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
        system_message: Optional[str] = None,
        return_logprobs: bool = False,
        return_logits: bool = False
    ):
        """Generate with harmony format from text prompt."""
        if self.harmony_renderer:
            harmony_formatted = create_harmony_template(
                prompt=prompt,
                system_message=system_message
            )
            prompt_tokens = self.tokenizer.encode(harmony_formatted)
        else:
            prompt_tokens = self.tokenizer.encode(prompt)
        
        yield from self.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            return_logprobs=return_logprobs,
            return_logits=return_logits,
            system_message=system_message
        )
