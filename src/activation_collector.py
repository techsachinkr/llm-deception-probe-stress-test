"""
Activation collection from Gemma 3 residual streams (§4.4).
Collects activations at every layer for every example,
at the last token of the generated response.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.utils import format_chat_prompt, save_pickle, load_pickle

logger = logging.getLogger(__name__)


class ActivationCollector:
    """
    Hooks into a model's residual stream to collect activations
    at every layer for specified token positions.
    
    Per §4.4: "We collect the residual stream activation at every layer
    at the last token of the generated response."
    """

    def __init__(self, model, tokenizer, n_layers: int = None, d_model: int = None):
        self.model = model
        self.tokenizer = tokenizer
        self._activations = {}
        self._hooks = []

        # Auto-detect architecture: try multiple attribute names
        model_config = getattr(model, "config", None)
        
        # Detect d_model
        if d_model is not None:
            self.d_model = d_model
        else:
            self.d_model = None
            for attr in ["hidden_size", "d_model", "n_embd", "hidden_dim", "model_dim"]:
                val = getattr(model_config, attr, None)
                if val is not None:
                    self.d_model = val
                    break
            # Try nested text_config (Gemma 3 multimodal models)
            if self.d_model is None:
                text_config = getattr(model_config, "text_config", None)
                if text_config is not None:
                    for attr in ["hidden_size", "d_model", "n_embd"]:
                        val = getattr(text_config, attr, None)
                        if val is not None:
                            self.d_model = val
                            break
        
        # Detect n_layers
        if n_layers is not None:
            self.n_layers = n_layers
        else:
            self.n_layers = None
            for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                val = getattr(model_config, attr, None)
                if val is not None:
                    self.n_layers = val
                    break
            if self.n_layers is None:
                text_config = getattr(model_config, "text_config", None)
                if text_config is not None:
                    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                        val = getattr(text_config, attr, None)
                        if val is not None:
                            self.n_layers = val
                            break

        # Fallback: count actual layers
        try:
            actual_layers = len(self._get_layer_modules())
            if self.n_layers is None:
                self.n_layers = actual_layers
            elif actual_layers != self.n_layers:
                logger.warning(f"Config says {self.n_layers} layers but model has {actual_layers}. Using {actual_layers}.")
                self.n_layers = actual_layers
        except Exception:
            pass

        # Fallback: probe model with a dummy forward pass to get d_model
        if self.d_model is None:
            logger.info("Probing model to detect hidden dimension...")
            try:
                dummy_input = tokenizer("test", return_tensors="pt")
                dummy_input = {k: v.to(model.device) for k, v in dummy_input.items()}
                layers = self._get_layer_modules()
                probe_result = [None]
                def _probe_hook(module, input, output):
                    if isinstance(output, tuple):
                        probe_result[0] = output[0].shape[-1]
                    else:
                        probe_result[0] = output.shape[-1]
                hook = layers[0].register_forward_hook(_probe_hook)
                with torch.no_grad():
                    model(**dummy_input)
                hook.remove()
                self.d_model = probe_result[0]
            except Exception as e:
                logger.error(f"Could not auto-detect d_model: {e}")
                raise ValueError("Cannot detect d_model. Pass it explicitly: ActivationCollector(model, tok, d_model=2560)")

        assert self.n_layers is not None, "Could not detect n_layers"
        assert self.d_model is not None, "Could not detect d_model"
        
        logger.info(f"ActivationCollector: n_layers={self.n_layers}, d_model={self.d_model}")

    def _get_layer_modules(self):
        """Get the residual stream output hook points for each layer.
        Uses recursive search to handle any model wrapping."""
        
        # First: try direct known paths
        candidates = []
        
        def _try_path(obj, attrs):
            """Navigate a dotted path like 'model.layers'."""
            for attr in attrs:
                obj = getattr(obj, attr, None)
                if obj is None:
                    return None
            return list(obj) if hasattr(obj, '__iter__') else None
        
        paths = [
            ["model", "layers"],
            ["model", "model", "layers"],
            ["language_model", "model", "layers"],
            ["transformer", "h"],
            ["model", "decoder", "layers"],
            ["text_model", "model", "layers"],
            ["text_model", "layers"],
            ["layers"],
        ]
        
        for path in paths:
            result = _try_path(self.model, path)
            if result and len(result) > 0:
                logger.debug(f"Found layers at model.{'.'.join(path)} ({len(result)} layers)")
                return result
        
        # Fallback: recursively search for any nn.ModuleList with the right length
        import torch.nn as nn
        found = []
        
        def _search(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.ModuleList) and len(child) > 10:
                    found.append((full_name, list(child)))
                _search(child, full_name)
        
        _search(self.model)
        
        if found:
            # Pick the ModuleList closest to expected n_layers
            if self.n_layers:
                best = min(found, key=lambda x: abs(len(x[1]) - self.n_layers))
            else:
                best = max(found, key=lambda x: len(x[1]))
            logger.info(f"Found layers via search: {best[0]} ({len(best[1])} layers)")
            return best[1]
        
        # Debug: print model structure to help diagnose
        logger.error("Could not find transformer layers. Model structure:")
        for name, child in self.model.named_children():
            logger.error(f"  model.{name}: {type(child).__name__}")
            for name2, child2 in child.named_children():
                logger.error(f"    model.{name}.{name2}: {type(child2).__name__}")
        
        raise ValueError("Cannot find transformer layers. See model structure above.")

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output is typically a tuple; first element is the hidden state
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Store on CPU to save GPU memory
            self._activations[layer_idx] = hidden.detach().cpu()
        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all layers."""
        self.remove_hooks()
        layers = self._get_layer_modules()
        assert len(layers) == self.n_layers, \
            f"Expected {self.n_layers} layers, found {len(layers)}"
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)
        logger.debug(f"Registered hooks on {len(self._hooks)} layers")

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()

    @torch.no_grad()
    def collect_single(
        self,
        prompt_text: str,
        response_text: str,
        system_prompt: str,
        use_neutral_context: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Collect activations for a single (prompt, response) pair.
        
        CRITICAL: use_neutral_context=True (default) feeds BOTH honest and
        deceptive responses through the same neutral system prompt during
        activation collection. This prevents the probe from trivially
        separating based on system prompt differences rather than
        response content. The original system prompts are only used
        during data GENERATION, not during activation collection.
        
        Returns:
            Dict with:
              "response_last": np.array of shape (n_layers, d_model)
              "prompt_last": np.array of shape (n_layers, d_model)
        """
        self._activations.clear()
        
        # Use a neutral system prompt for activation collection
        # so honest/deceptive activations differ only in response content
        if use_neutral_context:
            collection_prompt = "Read the following question and response."
        else:
            collection_prompt = system_prompt
        
        # Tokenize the full conversation (prompt + response)
        full_text = format_chat_prompt(
            self.tokenizer, prompt_text, collection_prompt, response_text
        )
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # Also tokenize just the prompt to find the boundary
        prompt_only = format_chat_prompt(
            self.tokenizer, prompt_text, collection_prompt
        )
        prompt_ids = self.tokenizer(
            prompt_only, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]
        total_len = input_ids.shape[1]
        
        # Forward pass (triggers all hooks)
        self.model(input_ids=input_ids, attention_mask=inputs["attention_mask"].to(self.model.device))
        
        # Extract activations at key positions
        response_last_idx = total_len - 1       # last token of response
        prompt_last_idx = prompt_len - 1         # last token of prompt
        
        # Detect actual d_model from first layer's output
        actual_d = self._activations[0].shape[-1]
        if actual_d != self.d_model:
            logger.debug(f"Actual hidden dim {actual_d} differs from config {self.d_model}, adapting.")
            self.d_model = actual_d
        
        result = {
            "response_last": np.zeros((self.n_layers, self.d_model), dtype=np.float32),
            "prompt_last": np.zeros((self.n_layers, self.d_model), dtype=np.float32),
        }
        
        for layer_idx in range(self.n_layers):
            h = self._activations[layer_idx]     # (1, seq_len, d_model)
            result["response_last"][layer_idx] = h[0, response_last_idx].float().numpy()
            result["prompt_last"][layer_idx] = h[0, prompt_last_idx].float().numpy()
        
        self._activations.clear()
        return result

    @torch.no_grad()
    def collect_full_sequence(
        self,
        prompt_text: str,
        response_text: str,
        system_prompt: str,
    ) -> Dict[str, np.ndarray]:
        """
        Collect full-sequence activations for entropy analysis.
        Returns activations at ALL token positions for layers in top half.
        Memory-intensive; use sparingly.
        """
        self._activations.clear()
        
        full_text = format_chat_prompt(
            self.tokenizer, prompt_text, system_prompt, response_text
        )
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        
        self.model(input_ids=input_ids, attention_mask=inputs["attention_mask"].to(self.model.device))
        
        # Only store top-half layers for entropy (§3.4 calibration caveat)
        min_layer = self.n_layers // 2
        seq_len = input_ids.shape[1]
        result = {}
        
        for layer_idx in range(min_layer, self.n_layers):
            h = self._activations[layer_idx]  # (1, seq_len, d_model)
            result[layer_idx] = h[0].float().numpy()  # (seq_len, d_model)
        
        self._activations.clear()
        return result


def collect_dataset_activations(
    collector: ActivationCollector,
    examples: list,
    model_size: str,
    output_dir: str,
    dataset_name: str,
    skip_existing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect activations for an entire dataset split.
    
    Returns:
        activations: np.array (n_examples, n_layers, d_model) -- response-last activations
        labels: np.array (n_examples,) -- 0=honest, 1=deceptive
        prompt_activations: np.array (n_examples, n_layers, d_model) -- prompt-last activations
    """
    save_dir = Path(output_dir) / model_size / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    acts_path = save_dir / "activations.npy"
    labels_path = save_dir / "labels.npy"
    prompt_acts_path = save_dir / "prompt_activations.npy"
    
    if skip_existing and acts_path.exists() and labels_path.exists():
        logger.info(f"Loading cached activations from {save_dir}")
        return (
            np.load(str(acts_path)),
            np.load(str(labels_path)),
            np.load(str(prompt_acts_path)) if prompt_acts_path.exists() else None,
        )
    
    n = len(examples)
    n_layers = collector.n_layers
    d_model = collector.d_model
    
    # Pre-allocate arrays: for each example, we have both honest and deceptive
    activations = np.zeros((n * 2, n_layers, d_model), dtype=np.float32)
    prompt_activations = np.zeros((n * 2, n_layers, d_model), dtype=np.float32)
    labels = np.zeros(n * 2, dtype=np.int32)
    
    collector.register_hooks()
    
    try:
        for i, ex in enumerate(tqdm(examples, desc=f"Collecting {dataset_name}/{model_size}")):
            # Honest activation (label=0)
            honest_result = collector.collect_single(
                ex.question, ex.honest_response, ex.system_prompt_honest
            )
            activations[2 * i] = honest_result["response_last"]
            prompt_activations[2 * i] = honest_result["prompt_last"]
            labels[2 * i] = 0
            
            # Deceptive activation (label=1)
            deceptive_result = collector.collect_single(
                ex.question, ex.deceptive_response, ex.system_prompt_deceptive
            )
            activations[2 * i + 1] = deceptive_result["response_last"]
            prompt_activations[2 * i + 1] = deceptive_result["prompt_last"]
            labels[2 * i + 1] = 1
            
            # Periodic save
            if (i + 1) % 100 == 0:
                logger.info(f"  Collected {i + 1}/{n} examples")
                torch.cuda.empty_cache()
    finally:
        collector.remove_hooks()
    
    # Save
    np.save(str(acts_path), activations)
    np.save(str(labels_path), labels)
    np.save(str(prompt_acts_path), prompt_activations)
    logger.info(f"Saved activations: {activations.shape} to {save_dir}")
    
    return activations, labels, prompt_activations
