"""
Shared utilities: logging, seeding, model loading, serialization.
"""
import os
import json
import random
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    cache_dir: str = "./cache",
):
    """Load a Gemma 3 model and tokenizer with optional quantization."""
    logger.info(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = dtype

    # Try loading — Gemma 3 ≥4B are multimodal and may need torchvision
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ModuleNotFoundError, RuntimeError, ImportError) as e:
        err_msg = str(e).lower()
        if "torchvision" in err_msg or "conditionalgeneration" in err_msg or "vision" in err_msg:
            logger.warning(f"Multimodal model load failed: {e}")
            
            # Strategy 1: Reinstall matching torchvision
            try:
                import subprocess, sys
                logger.info("Installing compatible torchvision...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--force-reinstall",
                     "torchvision", "--quiet", "--break-system-packages"],
                    timeout=300, stderr=subprocess.DEVNULL,
                )
                # Retry after install
                import importlib
                if "torchvision" in sys.modules:
                    del sys.modules["torchvision"]
                    # Also clear submodules
                    for k in list(sys.modules.keys()):
                        if k.startswith("torchvision"):
                            del sys.modules[k]
                model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                logger.info("Model loaded after torchvision fix")
            except Exception as e2:
                logger.warning(f"torchvision fix failed: {e2}. Loading text backbone directly...")
                # Strategy 2: Load full multimodal model via AutoModel and extract LM
                try:
                    # Suppress the torchvision import by mocking it
                    import types
                    mock_tv = types.ModuleType("torchvision")
                    mock_transforms = types.ModuleType("torchvision.transforms")
                    mock_transforms.InterpolationMode = type("InterpolationMode", (), {
                        "BILINEAR": 2, "BICUBIC": 3, "NEAREST": 0,
                    })
                    sys.modules["torchvision"] = mock_tv
                    sys.modules["torchvision.transforms"] = mock_transforms
                    # Also mock submodules that may be imported
                    for submod in ["_meta_registrations", "datasets", "io", "models", "ops", "utils"]:
                        sys.modules[f"torchvision.{submod}"] = types.ModuleType(f"torchvision.{submod}")
                    mock_tv.transforms = mock_transforms
                    
                    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                    logger.info("Model loaded with mocked torchvision")
                except Exception as e3:
                    logger.error(f"All loading strategies failed: {e3}")
                    raise RuntimeError(
                        f"Cannot load {model_id}. Fix torchvision: "
                        f"pip install torchvision --force-reinstall --break-system-packages"
                    ) from e3
        else:
            raise
    model.eval()
    logger.info(f"Model loaded: {model_id} ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")
    return model, tokenizer


def format_chat_prompt(
    tokenizer,
    question: str,
    system_prompt: str,
    response: Optional[str] = None,
) -> str:
    """Format a prompt using the model's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    # Try the tokenizer's chat template, fall back to manual formatting
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(response is None)
        )
    except Exception:
        # Manual fallback for Gemma-style formatting
        text = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
        text += f"<start_of_turn>user\n{question}<end_of_turn>\n"
        if response is not None:
            text += f"<start_of_turn>model\n{response}<end_of_turn>"
        else:
            text += "<start_of_turn>model\n"
    return text


def generate_response(
    model, tokenizer, prompt_text: str, max_new_tokens: int = 256,
    temperature: float = 0.7, do_sample: bool = True,
) -> str:
    """Generate a single response from the model. NaN-safe."""
    from transformers import LogitsProcessor, LogitsProcessorList
    
    class NaNSafeLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1e9))
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([NaNSafeLogitsProcessor()]),
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def save_pickle(obj: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def get_gpu_memory():
    """Report current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
