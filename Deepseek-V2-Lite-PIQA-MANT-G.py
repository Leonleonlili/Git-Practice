#!/usr/bin/env python3
"""
PIQA evaluation for DeepSeek-V2-Lite with M-ANT quantization.

The M-ANT quantization flow, conductance export/recovery logic, and PIQA prompt
and scoring style are kept aligned with SmolLM-360M-MLA-d_kv_16-PIQA-MANT-G.py.
This version only swaps model loading and checkpoint compatibility to work with
the local DeepSeek-V2-Lite checkpoint in ./DeepSeek-V2-Lite.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


TOKENIZATION_BOUNDARY_WARNING_EMITTED = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_name)


def resolve_torch_dtype(dtype_name: str, config: Any, target_device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        config_dtype = getattr(config, "torch_dtype", None)
        if isinstance(config_dtype, str):
            config_dtype = getattr(torch, config_dtype, None)
        if isinstance(config_dtype, torch.dtype):
            if target_device.type == "cpu" and config_dtype in {torch.float16, torch.bfloat16}:
                return torch.float32
            return config_dtype
        return torch.float32 if target_device.type == "cpu" else torch.bfloat16

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    runtime_dtype = mapping[dtype_name]
    if target_device.type == "cpu" and runtime_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return runtime_dtype


def get_model_input_device(model: Any) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    return next(model.parameters()).device


def patch_transformers_compatibility() -> None:
    try:
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    if not hasattr(import_utils, "is_torch_fx_available"):
        def is_torch_fx_available() -> bool:
            return hasattr(torch, "fx")

        import_utils.is_torch_fx_available = is_torch_fx_available


def load_module_from_file(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create a module spec for {file_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_deepseek_rope_scaling(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    rope_scaling = config_dict.get("rope_scaling")
    if not isinstance(rope_scaling, dict):
        return config_dict

    normalized = dict(rope_scaling)
    for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
        value = normalized.get(key)
        if isinstance(value, int):
            normalized[key] = float(value)

    config_dict = dict(config_dict)
    config_dict["rope_scaling"] = normalized
    return config_dict


def load_local_deepseek_classes(model_path: Path) -> Optional[Tuple[Any, Any]]:
    config_file = model_path / "configuration_deepseek.py"
    model_file = model_path / "modeling_deepseek.py"
    if not config_file.exists() or not model_file.exists():
        return None

    package_hash = hashlib.md5(str(model_path.resolve()).encode("utf-8")).hexdigest()[:12]
    package_name = f"_local_deepseek_{package_hash}"

    package_module = sys.modules.get(package_name)
    if package_module is None:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(model_path)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package_module

    config_module_name = f"{package_name}.configuration_deepseek"
    model_module_name = f"{package_name}.modeling_deepseek"

    config_module = sys.modules.get(config_module_name)
    if config_module is None:
        config_module = load_module_from_file(config_module_name, config_file)

    model_module = sys.modules.get(model_module_name)
    if model_module is None:
        model_module = load_module_from_file(model_module_name, model_file)

    config_class = getattr(config_module, "DeepseekV2Config", None)
    model_class = getattr(model_module, "DeepseekV2ForCausalLM", None)
    if config_class is None or model_class is None:
        raise AttributeError(
            "Local DeepSeek modules were loaded, but DeepseekV2Config or "
            "DeepseekV2ForCausalLM was not found."
        )

    return config_class, model_class


def load_deepseek_model(
    model_path: str,
    device: torch.device,
    dtype_name: str = "auto",
    device_map: str = "none",
) -> Tuple[Any, Any, Any]:
    model_dir = Path(model_path)

    print("\n" + "=" * 80)
    print("MODEL LOADING PROCESS")
    print("=" * 80)
    print(f"Model path: {model_dir}\n")

    patch_transformers_compatibility()
    local_classes = load_local_deepseek_classes(model_dir)
    if local_classes is not None:
        config_class, model_class = local_classes
        with (model_dir / "config.json").open("r", encoding="utf-8") as fin:
            config_dict = json.load(fin)
        config_dict = normalize_deepseek_rope_scaling(config_dict)
        config = config_class.from_dict(config_dict)
        config._name_or_path = str(model_dir)
    else:
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        model_class = None

    runtime_dtype = resolve_torch_dtype(dtype_name, config, device)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": runtime_dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map == "auto":
        load_kwargs["device_map"] = "auto"

    try:
        if model_class is not None:
            model = model_class.from_pretrained(str(model_dir), config=config, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    except Exception as exc:
        exc_message = str(exc).lower()
        can_retry_without_low_cpu_mem = (
            device_map == "none"
            and load_kwargs.get("low_cpu_mem_usage", False)
            and ("accelerate" in exc_message or "low_cpu_mem_usage" in exc_message)
        )
        if not can_retry_without_low_cpu_mem:
            raise

        print(
            "Warning: low_cpu_mem_usage loading failed. "
            "Retrying with the standard from_pretrained path."
        )
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs.pop("low_cpu_mem_usage", None)
        if model_class is not None:
            model = model_class.from_pretrained(str(model_dir), config=config, **fallback_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), **fallback_kwargs)

    if device_map == "none":
        model = model.to(device=device, dtype=runtime_dtype)

    model.eval()

    try:
        generation_config = GenerationConfig.from_pretrained(str(model_dir))
        generation_config.pad_token_id = generation_config.eos_token_id
        model.generation_config = generation_config
    except Exception:
        pass

    print("Model load summary:")
    print(f"  model_type     : {getattr(config, 'model_type', 'unknown')}")
    print(f"  architectures  : {getattr(config, 'architectures', None)}")
    print(f"  target_device  : {device}")
    print(f"  input_device   : {get_model_input_device(model)}")
    print(f"  device_map     : {device_map}")
    print(f"  runtime_dtype  : {next(model.parameters()).dtype}")
    print("=" * 80 + "\n")
    return model, tokenizer, config


class MANTQuantizerWithConductance:
    """
    M-ANT (Mathematically Adaptive Numerical Type) Quantizer
    Extended to save conductance states for RRAM simulation
    
    Formula: x_hat = s * sigma * (a * k + 2^k)
    where:
        - s: scaling factor
        - sigma: sign bit
        - a: shape coefficient (searched from candidates)
        - k: magnitude index (0 to 2^(bit_width-1) - 1)
    """
    
    def __init__(self, bit_width=4, group_size=128, G_MIN=20.0, G_MAX=200.0):
        """
        Args:
            bit_width: Total bits for quantization (1 bit for sign, rest for magnitude)
            group_size: Number of weights per quantization group
            G_MIN: Minimum conductance in μS (default: 20.0)
            G_MAX: Maximum conductance in μS (default: 200.0)
        """
        self.bit_width = bit_width
        self.group_size = group_size
        self.max_int = 2**(bit_width - 1) - 1  # Maximum index k
        
        # RRAM hardware parameters (in microsiemens, μS)
        self.G_MIN = G_MIN
        self.G_MAX = G_MAX
        self.G_RANGE = G_MAX - G_MIN
        
        # Shape coefficient search space
        self.a_candidates = [0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
        
        print(f"M-ANT Quantizer with Conductance Mapping initialized:")
        print(f"  Bit width: {bit_width} (1 sign bit + {bit_width-1} magnitude bits)")
        print(f"  Group size: {group_size}")
        print(f"  Max index k: {self.max_int}")
        print(f"  Shape coefficient candidates: {self.a_candidates}")
        print(f"  Conductance range: [{G_MIN}, {G_MAX}] μS")
    
    def get_mant_grid(self, a, scale, device):
        """
        Generate M-ANT quantization grid for positive values.
        
        Formula: grid[k] = scale * (a * k + 2^k)
        
        Args:
            a: Shape coefficient
            scale: Scaling factor
            
        Returns:
            Tensor of quantization levels
        """
        k = torch.arange(0, self.max_int + 1, dtype=torch.float32, device=device)
        # M-ANT formula: scale * (a*k + 2^k)
        grid = scale * (a * k + torch.pow(2, k))
        return grid
    
    def map_to_conductance_differential(self, w_abs, w_sign, grid, indices):
        """
        Map M-ANT quantized values to RRAM differential conductance pairs.
        
        This implements mapping to differential pair architecture:
        - For positive weights: G+ = mapped conductance, G- = G_MIN
        - For negative weights: G+ = G_MIN, G- = mapped conductance
        
        Args:
            w_abs: Absolute weight values
            w_sign: Sign of weights
            grid: M-ANT quantization grid (absolute values)
            indices: Quantization indices for each weight
            
        Returns:
            G_pos: Positive channel conductances (μS)
            G_neg: Negative channel conductances (μS)
        """
        # Get M-ANT quantized absolute values
        mant_values = grid[indices]
        
        # Normalize to [0, 1] for conductance mapping
        max_mant = grid.max()
        if max_mant == 0:
            return torch.full_like(w_abs, self.G_MIN), torch.full_like(w_abs, self.G_MIN)
        
        mant_norm = mant_values / max_mant
        
        # Map to conductance range [G_MIN, G_MAX]
        G_mapped = self.G_MIN + mant_norm * self.G_RANGE
        
        # Create differential pairs based on sign
        G_pos = torch.zeros_like(w_abs)
        G_neg = torch.zeros_like(w_abs)
        
        pos_mask = (w_sign >= 0)
        neg_mask = (w_sign < 0)
        
        # Positive weights: G+ carries the value, G- is baseline
        G_pos[pos_mask] = G_mapped[pos_mask]
        G_neg[pos_mask] = self.G_MIN
        
        # Negative weights: G- carries the value, G+ is baseline
        G_pos[neg_mask] = self.G_MIN
        G_neg[neg_mask] = G_mapped[neg_mask]
        
        return G_pos, G_neg
    
    def quantize_group(self, w_group, show_progress=False, group_idx=0, total_groups=0):
        """
        Quantize a single group by searching for the best shape coefficient 'a'.
        
        Now also returns conductance mapping information.
        
        Args:
            w_group: Weight group tensor (shape: [group_size])
            show_progress: Whether to show search progress
            group_idx: Current group index (for progress display)
            total_groups: Total number of groups (for progress display)
            
        Returns:
            dict containing:
                - 'weight': Quantized and reconstructed weight group
                - 'G_pos': Positive channel conductances (μS)
                - 'G_neg': Negative channel conductances (μS)
                - 'a': Best shape coefficient
                - 'scale': Scaling factor
                - 'w_max': Maximum absolute weight value in group
        """
        w_abs = w_group.abs()
        w_sign = w_group.sign()
        w_max = w_abs.max()
        
        # Handle edge case: all zeros
        if w_max == 0:
            return {
                'weight': torch.zeros_like(w_group),
                'G_pos': torch.full_like(w_group, self.G_MIN),
                'G_neg': torch.full_like(w_group, self.G_MIN),
                'a': 0,
                'scale': 0.0,
                'w_max': 0.0
            }
        
        best_mse = float('inf')
        best_result = None
        
        # Search for the best shape coefficient 'a'
        for a in self.a_candidates:
            # 1. Calculate scale factor to align max value with max grid point
            max_val_mant = (a * self.max_int + 2**self.max_int)
            
            if max_val_mant == 0:
                continue
            
            scale = w_max / max_val_mant
            
            # 2. Build M-ANT quantization grid
            grid = self.get_mant_grid(a, scale, w_group.device)
            
            # 3. Quantize: find nearest grid point for each weight
            diff = (w_abs.unsqueeze(-1) - grid).abs()
            indices = diff.argmin(dim=-1)
            
            # 4. Map to differential conductance pairs (IDEAL, no noise yet)
            G_pos, G_neg = self.map_to_conductance_differential(w_abs, w_sign, grid, indices)
            
            # 5. Dequantize: reconstruct weights from IDEAL conductances
            # This simulates the ideal case (no RRAM non-idealities)
            # G_diff = G_pos - G_neg is already the differential (no G_MIN offset)
            # For positive: G_diff = (G_MIN + mant_norm*G_RANGE) - G_MIN = mant_norm*G_RANGE
            # For negative: G_diff = G_MIN - (G_MIN + mant_norm*G_RANGE) = -mant_norm*G_RANGE
            G_diff = G_pos - G_neg
            
            # Reconstruct: w = G_diff / G_RANGE * w_max
            w_hat = G_diff / self.G_RANGE * w_max
            
            # 6. Compute MSE
            mse = torch.mean((w_group - w_hat)**2).item()
            
            # 7. Track best result
            if mse < best_mse:
                best_mse = mse
                best_result = {
                    'weight': w_hat,
                    'G_pos': G_pos,
                    'G_neg': G_neg,
                    'a': a,
                    'scale': scale.item() if isinstance(scale, torch.Tensor) else scale,
                    'w_max': w_max.item() if isinstance(w_max, torch.Tensor) else w_max
                }
        
        return best_result
    
    def quantize_tensor(self, weight_tensor, layer_name="", show_progress=True):
        """
        Quantize an entire weight tensor using group-wise M-ANT quantization.
        
        Now also returns conductance mapping information for each group.
        
        Args:
            weight_tensor: Original weight tensor (any shape)
            layer_name: Name of the layer (for progress display)
            show_progress: Whether to show detailed progress
            
        Returns:
            dict containing:
                - 'weight': Quantized weight tensor (same shape as input)
                - 'conductance_info': Dict with conductance mapping for each group
        """
        original_shape = weight_tensor.shape
        device = weight_tensor.device
        dtype = weight_tensor.dtype
        
        # Flatten and pad to group_size
        W_flat = weight_tensor.flatten()
        num_elements = W_flat.numel()
        
        # Pad if necessary
        pad_size = 0
        if num_elements % self.group_size != 0:
            pad_size = self.group_size - (num_elements % self.group_size)
            W_flat = torch.cat([W_flat, torch.zeros(pad_size, device=device, dtype=dtype)])
        
        # Reshape into groups
        W_groups = W_flat.view(-1, self.group_size)
        num_groups = W_groups.shape[0]
        
        if show_progress:
            print(f"    └─ Quantizing {num_groups} groups (saving conductances)...", end='', flush=True)
        
        # Quantize each group and collect conductance info
        W_quantized = torch.zeros_like(W_groups)
        G_pos_all = torch.zeros_like(W_groups)
        G_neg_all = torch.zeros_like(W_groups)
        a_list = []
        scale_list = []
        w_max_list = []
        
        # Show progress every 10% or at least every 100 groups
        progress_interval = max(1, min(100, num_groups // 10))
        
        for i in range(num_groups):
            result = self.quantize_group(W_groups[i], show_progress=False, 
                                        group_idx=i, total_groups=num_groups)
            
            W_quantized[i] = result['weight']
            G_pos_all[i] = result['G_pos']
            G_neg_all[i] = result['G_neg']
            a_list.append(result['a'])
            scale_list.append(result['scale'])
            w_max_list.append(result['w_max'])
            
            # Show progress
            if show_progress and (i + 1) % progress_interval == 0:
                progress_pct = (i + 1) / num_groups * 100
                print(f"\r    └─ Quantizing {num_groups} groups (saving conductances)... {progress_pct:.0f}%", end='', flush=True)
        
        if show_progress:
            print(f"\r    └─ Quantizing {num_groups} groups (saving conductances)... 100% ✓")
        
        # Flatten and remove padding
        W_quantized_flat = W_quantized.flatten()[:num_elements]
        G_pos_flat = G_pos_all.flatten()[:num_elements]
        G_neg_flat = G_neg_all.flatten()[:num_elements]
        
        # Reshape to original shape
        W_quantized_reshaped = W_quantized_flat.view(original_shape)
        G_pos_reshaped = G_pos_flat.view(original_shape)
        G_neg_reshaped = G_neg_flat.view(original_shape)
        
        # Prepare conductance info
        conductance_info = {
            'G_pos': G_pos_reshaped,  # Positive channel conductances
            'G_neg': G_neg_reshaped,  # Negative channel conductances
            'a_per_group': a_list,     # Best 'a' for each group
            'scale_per_group': scale_list,  # Scale factor for each group
            'w_max_per_group': w_max_list,  # Max weight for each group
            'num_groups': num_groups,
            'group_size': self.group_size,
            'original_shape': original_shape,
            'num_padded': pad_size
        }
        
        return {
            'weight': W_quantized_reshaped,
            'conductance_info': conductance_info
        }


def save_quantized_model_with_conductance(model, conductance_map, save_path, bit_width, 
                                          group_size, original_model_path, G_MIN=20.0, G_MAX=200.0):
    """
    Save quantized model weights AND conductance mappings to disk.
    
    Args:
        model: Quantized model
        conductance_map: Dict mapping layer names to conductance info
        save_path: Directory to save the quantized weights and conductances
        bit_width: Bit width used for quantization
        group_size: Group size used for quantization
        original_model_path: Path to original model (for reference)
        G_MIN: Minimum conductance (μS)
        G_MAX: Maximum conductance (μS)
    """
    print("\n" + "="*80)
    print("SAVING QUANTIZED MODEL WITH CONDUCTANCE MAPPING")
    print("="*80)
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save quantization config
    quant_config = {
        "quantization_method": "M-ANT with Conductance Mapping",
        "bit_width": bit_width,
        "group_size": group_size,
        "rram_config": {
            "G_min_uS": G_MIN,
            "G_max_uS": G_MAX,
            "architecture": "differential_pair"
        },
        "original_model_path": original_model_path,
        "quantization_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    config_file = os.path.join(save_path, "quantization_config.json")
    with open(config_file, 'w') as f:
        json.dump(quant_config, f, indent=2)
    print(f"✓ Saved quantization config to: {config_file}")
    
    # Save model weights (quantized, dequantized back to FP)
    print("Collecting model state dict...")
    state_dict = model.state_dict()
    
    print("Moving tensors to CPU...")
    state_dict_cpu = {k: v.cpu().contiguous() for k, v in state_dict.items()}
    
    weights_file = os.path.join(save_path, "model.safetensors")
    print(f"Saving weights to: {weights_file}")
    save_file(state_dict_cpu, weights_file)
    print(f"✓ Saved quantized weights: {weights_file}")
    
    # Save conductance mapping (the key addition!)
    print("\nSaving conductance mappings...")
    conductance_file = os.path.join(save_path, "conductance_mapping.pkl")
    
    # Convert tensors to CPU for saving
    conductance_map_cpu = {}
    for layer_name, cond_info in conductance_map.items():
        conductance_map_cpu[layer_name] = {
            'G_pos': cond_info['G_pos'].cpu().contiguous(),
            'G_neg': cond_info['G_neg'].cpu().contiguous(),
            'a_per_group': cond_info['a_per_group'],
            'scale_per_group': cond_info['scale_per_group'],
            'w_max_per_group': cond_info['w_max_per_group'],
            'num_groups': cond_info['num_groups'],
            'group_size': cond_info['group_size'],
            'original_shape': cond_info['original_shape'],
            'num_padded': cond_info['num_padded']
        }
    
    with open(conductance_file, 'wb') as f:
        pickle.dump(conductance_map_cpu, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved conductance mapping to: {conductance_file}")
    
    # Print statistics
    total_conductances = sum(info['G_pos'].numel() for info in conductance_map.values())
    print(f"\n  Conductance Mapping Statistics:")
    print(f"    Total layers with conductances: {len(conductance_map)}")
    print(f"    Total conductance pairs: {total_conductances:,}")
    print(f"    File size: {os.path.getsize(conductance_file) / (1024**2):.2f} MB")
    
    # Copy config files from original model
    print("\nCopying config files from original model...")
    for config_filename in [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "configuration_deepseek.py",
        "modeling_deepseek.py",
        "tokenization_deepseek_fast.py",
        "README.md",
        "LICENSE",
        ".gitattributes",
    ]:
        src = os.path.join(original_model_path, config_filename)
        dst = os.path.join(save_path, config_filename)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {config_filename}")
    
    print("\n" + "="*80)
    print(f"QUANTIZED MODEL + CONDUCTANCE MAPPING SAVED SUCCESSFULLY!")
    print(f"Location: {save_path}")
    print(f"  - Quantized weights: model.safetensors")
    print(f"  - Conductance mapping: conductance_mapping.pkl")
    print(f"  - Config: quantization_config.json")
    print("="*80 + "\n")


def recover_weights_from_conductance(conductance_map, G_MIN=20.0, G_MAX=200.0):
    """
    Recover weights from conductance differential pairs.
    
    This simulates the ideal recovery process (no RRAM non-idealities).
    
    Args:
        conductance_map: Dict mapping layer names to conductance info
        G_MIN: Minimum conductance (μS)
        G_MAX: Maximum conductance (μS)
        
    Returns:
        state_dict: Dict mapping layer names to recovered weight tensors
    """
    print("\n" + "="*80)
    print("RECOVERING WEIGHTS FROM CONDUCTANCE STATES")
    print("="*80)
    print(f"Conductance range: [{G_MIN}, {G_MAX}] μS")
    print(f"Total layers to recover: {len(conductance_map)}\n")
    
    state_dict = {}
    G_RANGE = G_MAX - G_MIN
    
    for layer_name, cond_info in tqdm(conductance_map.items(), desc="Recovering weights"):
        G_pos = cond_info['G_pos']
        G_neg = cond_info['G_neg']
        w_max_per_group = cond_info['w_max_per_group']
        original_shape = cond_info['original_shape']
        group_size = cond_info['group_size']
        num_groups = cond_info['num_groups']
        num_padded = cond_info['num_padded']
        
        # Flatten conductances
        G_pos_flat = G_pos.flatten()
        G_neg_flat = G_neg.flatten()
        
        # Recover weights group by group
        num_elements = G_pos_flat.numel() - num_padded
        W_recovered = torch.zeros(G_pos_flat.numel(), device=G_pos_flat.device)
        
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = min(start_idx + group_size, num_elements)
            
            if start_idx >= num_elements:
                break
            
            # Get conductances for this group
            G_p_group = G_pos_flat[start_idx:end_idx]
            G_n_group = G_neg_flat[start_idx:end_idx]
            w_max = w_max_per_group[group_idx]
            
            # Recovery formula: w = (G+ - G-) / G_RANGE * w_max
            G_diff = G_p_group - G_n_group
            
            # Handle the baseline offset
            # Since we stored: G_mapped = G_MIN + norm * G_RANGE
            # We need: w = (G_diff - G_MIN) / G_RANGE * w_max (accounting for baseline)
            # But actually G_diff already contains the differential, so:
            w_group = G_diff / G_RANGE * w_max
            
            W_recovered[start_idx:end_idx] = w_group
        
        # Remove padding and reshape
        W_recovered = W_recovered[:num_elements].view(original_shape)
        
        # Store with correct key name (add .weight suffix if not present)
        param_name = layer_name if layer_name.endswith('.weight') else f"{layer_name}.weight"
        state_dict[param_name] = W_recovered
    
    print("\n" + "="*80)
    print("WEIGHT RECOVERY COMPLETED!")
    print(f"Recovered {len(state_dict)} weight tensors")
    print("="*80 + "\n")
    
    return state_dict


def load_quantized_model_weights(model, quantized_model_path, device="cuda", 
                                 use_conductance_recovery=False):
    """
    Load pre-quantized weights into model.
    
    Args:
        model: Model with correct architecture
        quantized_model_path: Path to quantized weights
        device: Device to load weights to
        use_conductance_recovery: If True, recover weights from conductance mapping
        
    Returns:
        model: Model with quantized weights loaded
        quant_config: Quantization configuration
    """
    print("\n" + "="*80)
    print("LOADING PRE-QUANTIZED MODEL")
    print("="*80)
    
    # Load quantization config
    config_file = os.path.join(quantized_model_path, "quantization_config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Quantization config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        quant_config = json.load(f)
    
    print(f"Quantization method: {quant_config['quantization_method']}")
    print(f"Bit width: {quant_config['bit_width']}")
    print(f"Group size: {quant_config['group_size']}")
    print(f"Quantized at: {quant_config['quantization_time']}")
    
    if use_conductance_recovery:
        print(f"Mode: Recover weights from conductance states")
        
        # Load conductance mapping
        conductance_file = os.path.join(quantized_model_path, "conductance_mapping.pkl")
        if not os.path.exists(conductance_file):
            raise FileNotFoundError(f"Conductance mapping not found: {conductance_file}")
        
        print(f"\nLoading conductance mapping from: {conductance_file}")
        with open(conductance_file, 'rb') as f:
            conductance_map = pickle.load(f)
        print(f"  ✓ Loaded conductance mapping for {len(conductance_map)} layers")
        
        # Get RRAM config
        G_MIN = quant_config.get('rram_config', {}).get('G_min_uS', 20.0)
        G_MAX = quant_config.get('rram_config', {}).get('G_max_uS', 200.0)
        
        # Recover weights from conductance states
        state_dict = recover_weights_from_conductance(conductance_map, G_MIN, G_MAX)
        
    else:
        print(f"Mode: Load directly from saved weights")
        print()
        
        # Load weights directly
        weights_file = os.path.join(quantized_model_path, "model.safetensors")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Quantized weights not found: {weights_file}")
        
        print(f"Loading weights from: {weights_file}")
        state_dict = load_file(weights_file)
    
    print("\nLoading state dict into model...")
    result = model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Weights loaded")
    print(f"    Missing keys: {len(result.missing_keys)}")
    print(f"    Unexpected keys: {len(result.unexpected_keys)}")
    
    print("\n" + "="*80)
    print("PRE-QUANTIZED MODEL LOADED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    return model, quant_config


def verify_quantization(model, bit_width=4, num_layers_to_check=3):
    """
    Verify that the model weights have been actually quantized.
    Check the number of unique values in weight tensors.
    """
    print("\n" + "="*80)
    print("QUANTIZATION VERIFICATION")
    print("="*80)
    
    linear_layers = [(name, module) for name, module in model.named_modules() 
                     if isinstance(module, nn.Linear)]
    
    # Check a few layers
    layers_to_check = min(num_layers_to_check, len(linear_layers))
    
    print(f"Checking {layers_to_check} layers for quantization effects...\n")
    
    for i in range(layers_to_check):
        name, module = linear_layers[i]
        weight = module.weight.data.float()
        
        # Count unique values
        unique_vals = torch.unique(weight)
        num_unique = len(unique_vals)
        
        expected_max = 2**bit_width
        
        print(f"Layer {i+1}: {name}")
        print(f"  Shape: {tuple(weight.shape)}")
        print(f"  Unique values: {num_unique}")
        print(f"  Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
        print(f"  Expected ~{expected_max} values per group (actual groups may vary)")
        
        # Sample some values to show they're discrete
        sample_vals = torch.sort(unique_vals)[0][:20].cpu().numpy()
        print(f"  Sample values: {sample_vals}")
        print()
    
    print("="*80 + "\n")


def quantize_model_mant(model, bit_width=4, group_size=128, G_MIN=20.0, G_MAX=200.0):
    """
    Apply M-ANT quantization to all linear layers in the model.
    Also collects conductance mappings for each layer.
    
    Args:
        model: PyTorch model
        bit_width: Bit width for quantization (INT4=4, INT5=5, INT6=6, INT8=8)
        group_size: Group size for quantization
        G_MIN: Minimum conductance (μS)
        G_MAX: Maximum conductance (μS)
        
    Returns:
        conductance_map: Dict mapping layer names to conductance info
    """
    print("\n" + "="*80)
    print(f"Starting M-ANT Quantization (INT{bit_width}) with Conductance Mapping")
    print("="*80)
    
    quantizer = MANTQuantizerWithConductance(bit_width=bit_width, group_size=group_size, 
                                             G_MIN=G_MIN, G_MAX=G_MAX)
    
    # Collect all linear layers
    print("\n[Step 1/3] Scanning model for linear layers...")
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    total_layers = len(linear_layers)
    print(f"  ✓ Found {total_layers} linear layers to quantize")
    
    # Calculate total parameters
    total_params = sum(module.weight.numel() for _, module in linear_layers)
    total_groups = sum((module.weight.numel() + group_size - 1) // group_size 
                       for _, module in linear_layers)
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Total groups: {total_groups:,}")
    
    print(f"\n[Step 2/3] Quantizing layers and mapping to conductances...")
    print("-" * 80)
    
    # Dictionary to store conductance mappings
    conductance_map = {}
    
    # Quantize each linear layer
    for idx, (name, module) in enumerate(linear_layers):
        # Get original weight
        original_weight = module.weight.data
        original_device = original_weight.device
        original_dtype = original_weight.dtype
        
        # Calculate layer info
        layer_params = original_weight.numel()
        layer_groups = (layer_params + group_size - 1) // group_size
        layer_shape = tuple(original_weight.shape)
        
        # Print layer info
        print(f"  [{idx+1}/{total_layers}] {name}")
        print(f"    ├─ Shape: {layer_shape}, Params: {layer_params:,}, Groups: {layer_groups}")
        
        # Quantize on the tensor's current device to support CPU, single-GPU,
        # or distributed device_map loading.
        weight_float = original_weight.float()
        
        # Apply M-ANT quantization and get conductance info
        result = quantizer.quantize_tensor(weight_float, layer_name=name, show_progress=True)
        quantized_weight = result['weight']
        conductance_info = result['conductance_info']
        
        # Store conductance mapping
        conductance_map[name] = conductance_info
        
        # Move back to original device and dtype
        quantized_weight = quantized_weight.to(device=original_device, dtype=original_dtype)
        
        # Replace model weight
        module.weight.data = quantized_weight
        
        # Verify quantization
        if idx == 0:
            unique_vals = torch.unique(quantized_weight.float())
            print(f"    ├─ VERIFICATION: Unique values in quantized weight: {len(unique_vals)}")
            print(f"    │  Expected for INT{bit_width}: ~{2**bit_width} values per group")
            print(f"    │  Weight range: [{quantized_weight.min().item():.6f}, {quantized_weight.max().item():.6f}]")
            
            # Verify conductance ranges
            G_pos = conductance_info['G_pos']
            G_neg = conductance_info['G_neg']
            print(f"    │  G+ range: [{G_pos.min().item():.2f}, {G_pos.max().item():.2f}] μS")
            print(f"    │  G- range: [{G_neg.min().item():.2f}, {G_neg.max().item():.2f}] μS")
        
        # Show progress
        overall_progress = (idx + 1) / total_layers * 100
        print(f"    └─ Overall progress: {overall_progress:.1f}% ({idx+1}/{total_layers} layers)")
        if idx < total_layers - 1:
            print()
    
    print("-" * 80)
    print(f"[Step 3/3] Finalizing...")
    print("  ✓ All layers quantized successfully!")
    print(f"  ✓ Conductance mappings collected for {len(conductance_map)} layers")
    
    print("\n" + "="*80)
    print("M-ANT Quantization with Conductance Mapping Completed!")
    print("="*80 + "\n")
    
    return conductance_map


def load_mla_model(model_path, device="cuda"):
    """Load MLA model from refactored checkpoint."""
    print(f"\n{'='*80}")
    print(f"MODEL LOADING PROCESS")
    print(f"{'='*80}")
    print(f"Model path: {model_path}\n")
    
    print("[Step 1/6] Loading config and tokenizer...")
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Config and tokenizer loaded\n")
    
    # Load checkpoint
    print("[Step 2/6] Loading checkpoint weights...")
    checkpoint = load_file(f"{model_path}/model.safetensors")
    print(f"  ✓ Loaded {len(checkpoint)} weight tensors\n")
    
    # Get dimensions from checkpoint
    print("[Step 3/6] Extracting MLA dimensions...")
    k_r_shape = checkpoint['model.layers.0.self_attn.k_r_proj.weight'].shape
    down_kv_shape = checkpoint['model.layers.0.self_attn.kv_proj.down_kv.weight'].shape
    up_k_shape = checkpoint['model.layers.0.self_attn.kv_proj.up_k.weight'].shape
    up_v_shape = checkpoint['model.layers.0.self_attn.kv_proj.up_v.weight'].shape
    
    print(f"  MLA checkpoint dimensions:")
    print(f"    k_r_proj: {k_r_shape}")
    print(f"    kv_proj.down_kv: {down_kv_shape}")
    print(f"    kv_proj.up_k: {up_k_shape}")
    print(f"    kv_proj.up_v: {up_v_shape}")
    print("  ✓ Dimensions extracted\n")
    
    # Create empty model
    print("[Step 4/6] Creating model architecture...")
    model = AutoModelForCausalLM.from_config(config)
    
    # Replace attention structure
    print(f"  Replacing attention structure for {config.num_hidden_layers} layers...")
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        
        # Add k_r_proj
        layer.self_attn.k_r_proj = nn.Linear(k_r_shape[1], k_r_shape[0], bias=False)
        
        # Add kv_proj (using LowRankKVLinear)
        layer.self_attn.kv_proj = LowRankKVLinear(
            d_in=down_kv_shape[1],
            d_k_out=up_k_shape[0],
            d_v_out=up_v_shape[0],
            d_mid=down_kv_shape[0],
            k_approx=True,
            v_approx=True,
            kv_joint=True,
            bias=False
        )
        
        # Remove original k_proj and v_proj
        delattr(layer.self_attn, 'k_proj')
        delattr(layer.self_attn, 'v_proj')
        
        # Show progress every 25%
        if (layer_idx + 1) % max(1, config.num_hidden_layers // 4) == 0:
            progress = (layer_idx + 1) / config.num_hidden_layers * 100
            print(f"    {progress:.0f}% complete ({layer_idx + 1}/{config.num_hidden_layers} layers)")
    print("  ✓ Architecture modified\n")
    
    # Load checkpoint
    print("[Step 5/6] Loading state dict into model...")
    result = model.load_state_dict(checkpoint, strict=False)
    print(f"  ✓ State dict loaded")
    print(f"    Missing keys: {len(result.missing_keys)}")
    print(f"    Unexpected keys: {len(result.unexpected_keys)}\n")
    
    # Generate q_idx and k_idx using parameters from config
    print("[Step 6/6] Generating RoPE indices and applying MLA patches...")
    mha2mla_args = types.SimpleNamespace(
        partial_rope_version='2-norm',
        top_k_rope_dim=8,  # SmolLM-360M uses 8
        last_k_rope_dim=0,
        rope_dim_for_mla=8,  # SmolLM-360M uses 8
        qk_tensor_path='./MHA2MLA/utils/smollm1_360M-2_norm_rank.pth',
        is_gqa2mha2mla=False,
    )
    
    q_masks, k_masks = partial_rope_mask(config, mha2mla_args)
    
    q_idx_list = []
    k_idx_list = []
    for layer_idx in range(config.num_hidden_layers):
        q_mask = q_masks[layer_idx]
        k_mask = k_masks[layer_idx]
        
        q_indices = reorder_matrix_rows(q_mask, is_cat=True)
        k_r_indices, _ = reorder_matrix_rows(k_mask, is_cat=False)
        
        n_head = config.num_attention_heads
        d_q_r = n_head * mha2mla_args.rope_dim_for_mla
        q_idx_list.append(q_indices[:d_q_r])
        k_idx_list.append(k_r_indices)
    
    print(f"  ✓ Generated indices: q_idx[0]={len(q_idx_list[0])}, k_idx[0]={len(k_idx_list[0])}")
    
    print(f"  Applying forward pass monkey patch...")
    mha2mla_llama(q_idx_list, k_idx_list)
    print(f"  ✓ Monkey patch applied\n")
    
    print(f"  Moving model to {device} with dtype bfloat16...")
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"  ✓ Model ready on {device}\n")
    
    print(f"{'='*80}")
    print("MODEL LOADED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    return model, tokenizer


def compute_choice_loglikelihood(model, tokenizer, prompt, choice_text, device="cuda"):
    """
    Compute log-likelihood of a choice given the prompt.
    Only evaluates the choice tokens, not the prompt.
    """
    # Tokenize prompt and full text separately
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    full_text = prompt + choice_text
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    
    # Get the choice tokens (everything after the prompt)
    choice_ids = full_ids[:, prompt_ids.shape[1]:]
    
    if choice_ids.shape[1] == 0:
        return float('-inf')
    
    # Compute log probabilities for the choice tokens only
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits
        
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Sum log probabilities for the choice tokens
        total_log_prob = 0.0
        for i, token_id in enumerate(choice_ids[0]):
            pos = prompt_ids.shape[1] + i - 1
            total_log_prob += log_probs[0, pos, token_id].item()
    
    # Normalize by number of choice tokens
    num_tokens = choice_ids.shape[1]
    normalized_score = total_log_prob / num_tokens if num_tokens > 0 else total_log_prob
    
    return normalized_score


def score_piqa_example(model, tokenizer, goal, sol1, sol2, device="cuda"):
    """Score a PIQA example and return the predicted solution."""
    prompt = f"Question: {goal}\nAnswer:"
    score1 = compute_choice_loglikelihood(model, tokenizer, prompt, f" {sol1}", device)
    score2 = compute_choice_loglikelihood(model, tokenizer, prompt, f" {sol2}", device)
    return 0 if score1 > score2 else 1


def load_piqa_local(piqa_dir="./PIQA"):
    """Load PIQA dataset from local directory."""
    data_file = os.path.join(piqa_dir, "dev.jsonl")
    labels_file = os.path.join(piqa_dir, "dev-labels.lst")
    
    examples = []
    with open(data_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    # Combine data and labels
    dataset = []
    for example, label in zip(examples, labels):
        dataset.append({
            'goal': example['goal'],
            'sol1': example['sol1'],
            'sol2': example['sol2'],
            'label': label
        })
    
    return dataset


def evaluate_piqa(model, tokenizer, device="cuda", num_samples=None):
    """Evaluate model on PIQA dataset."""
    print("Loading PIQA dataset from local files...")
    dataset = load_piqa_local("./PIQA")
    
    if num_samples:
        dataset = dataset[:num_samples]
    
    total_samples = len(dataset)
    print(f"Evaluating on {total_samples} examples...")
    
    correct = 0
    total = 0
    
    # Calculate checkpoint interval (every 5%)
    checkpoint_interval = max(1, int(total_samples * 0.05))
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating PIQA")):
        goal = example["goal"]
        sol1 = example["sol1"]
        sol2 = example["sol2"]
        label = example["label"]
        
        prediction = score_piqa_example(model, tokenizer, goal, sol1, sol2, device)
        
        if prediction == label:
            correct += 1
        total += 1
        
        # Print progress at 5% intervals
        if (idx + 1) % checkpoint_interval == 0:
            current_acc = correct / total
            progress_pct = (idx + 1) / total_samples * 100
            print(f"\n[Progress: {progress_pct:.1f}%] Current Accuracy: {current_acc:.4f} ({correct}/{total})")
    
    accuracy = correct / total
    print(f"\n" + "="*80)
    print(f"FINAL RESULTS:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*80)
    
    return accuracy


def load_mla_model(model_path, device="cuda", dtype_name="auto", device_map="none"):
    """Compatibility wrapper kept to minimize downstream changes."""
    if not isinstance(device, torch.device):
        device = resolve_device(device)
    model, tokenizer, _ = load_deepseek_model(
        model_path=model_path,
        device=device,
        dtype_name=dtype_name,
        device_map=device_map,
    )
    return model, tokenizer


def resolve_prefix_length(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    global TOKENIZATION_BOUNDARY_WARNING_EMITTED

    prompt_len = len(prompt_ids)
    if list(full_ids[:prompt_len]) == list(prompt_ids):
        return prompt_len

    prefix_len = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        prefix_len += 1

    if prefix_len == 0:
        raise ValueError("Could not align prompt tokens with prompt+answer tokens.")

    if not TOKENIZATION_BOUNDARY_WARNING_EMITTED:
        warnings.warn(
            "Prompt tokens were not a strict prefix of prompt+answer tokens. "
            "The scoring boundary will use the longest common prefix.",
            stacklevel=2,
        )
        TOKENIZATION_BOUNDARY_WARNING_EMITTED = True

    return prefix_len


def compute_choice_loglikelihood(model, tokenizer, prompt, choice_text, device="cuda"):
    """
    Compute normalized log-likelihood of a choice given the prompt.
    Only the choice tokens contribute to the final score.
    """
    del device
    input_device = get_model_input_device(model)

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + choice_text, add_special_tokens=True)["input_ids"]
    prefix_len = resolve_prefix_length(prompt_ids, full_ids)
    choice_len = len(full_ids) - prefix_len

    if choice_len <= 0:
        return float("-inf")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        shift_logits = outputs.logits[:, :-1, :].float()
        logits_device = shift_logits.device
        shift_labels = input_ids[:, 1:].to(logits_device)
        token_log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = token_log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

    start_idx = prefix_len - 1
    end_idx = start_idx + choice_len
    total_log_prob = float(token_log_probs[0, start_idx:end_idx].sum().item())
    return total_log_prob / float(choice_len)


def score_piqa_example(model, tokenizer, goal, sol1, sol2, device="cuda", answer_prefix=" "):
    """Score a PIQA example and return the predicted solution."""
    prompt = f"Question: {goal}\nAnswer:"
    score1 = compute_choice_loglikelihood(model, tokenizer, prompt, f"{answer_prefix}{sol1}", device)
    score2 = compute_choice_loglikelihood(model, tokenizer, prompt, f"{answer_prefix}{sol2}", device)
    return 0 if score1 > score2 else 1


def evaluate_piqa(model, tokenizer, device="cuda", num_samples=None, piqa_dir="./PIQA", answer_prefix=" "):
    """Evaluate model on PIQA using the original prompt and normalized choice scoring."""
    print("Loading PIQA dataset from local files...")
    dataset = load_piqa_local(piqa_dir)

    if num_samples is not None:
        dataset = dataset[:num_samples]

    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("PIQA dataset is empty.")

    print(f"Evaluating on {total_samples} examples...")

    correct = 0
    total = 0
    checkpoint_interval = max(1, int(total_samples * 0.05))

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating PIQA")):
        prediction = score_piqa_example(
            model,
            tokenizer,
            example["goal"],
            example["sol1"],
            example["sol2"],
            device,
            answer_prefix=answer_prefix,
        )

        if prediction == example["label"]:
            correct += 1
        total += 1

        if (idx + 1) % checkpoint_interval == 0:
            current_acc = correct / total
            progress_pct = (idx + 1) / total_samples * 100
            print(f"\n[Progress: {progress_pct:.1f}%] Current Accuracy: {current_acc:.4f} ({correct}/{total})")

    accuracy = correct / total
    print(f"\n" + "=" * 80)
    print("FINAL RESULTS:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 80)
    return accuracy


if False and __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate SmolLM-360M-MLA with M-ANT Quantization (with Conductance Mapping) on PIQA')
    parser.add_argument('--bit_width', type=int, default=4, 
                        help='Bit width for quantization (default: 4 for INT4)')
    parser.add_argument('--group_size', type=int, default=128,
                        help='Group size for quantization (default: 128)')
    parser.add_argument('--G_min', type=float, default=20.0,
                        help='Minimum conductance in μS (default: 20.0)')
    parser.add_argument('--G_max', type=float, default=200.0,
                        help='Maximum conductance in μS (default: 200.0)')
    parser.add_argument('--model_path', type=str, default='./SmolLM-360M-MLA-d_kv_16-refactor',
                        help='Path to model checkpoint')
    parser.add_argument('--quantized_model_path', type=str, default=None,
                        help='Path to pre-quantized model (skip quantization if provided)')
    parser.add_argument('--save_quantized', type=str, default='./SmolLM-360M-MLA-d_kv_16-MANT-INT4-G',
                        help='Path to save quantized model weights and conductances')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation (only quantize and save)')
    parser.add_argument('--use_conductance_recovery', action='store_true',
                        help='When evaluating, recover weights from conductance states (default: use direct weights)')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "=" * 80)
    print(f"SmolLM-360M-MLA M-ANT-INT{args.bit_width} with Conductance Mapping")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {device}")
    print(f"  Quantization: INT{args.bit_width} (M-ANT)")
    print(f"  Group size: {args.group_size}")
    print(f"  Conductance range: [{args.G_min}, {args.G_max}] μS")
    print(f"  Pre-quantized model: {args.quantized_model_path if args.quantized_model_path else 'None (will quantize)'}")
    print(f"  Save quantized: {args.save_quantized if args.save_quantized else 'No'}")
    print(f"  Skip evaluation: {args.skip_evaluation}")
    if not args.skip_evaluation:
        print(f"  Use conductance recovery: {args.use_conductance_recovery}")
    print(f"  Samples: {args.num_samples if args.num_samples else 'All'}")
    print("=" * 80 + "\n")
    
    # Load model
    model, tokenizer = load_mla_model(args.model_path, device)
    
    # Check if we should load pre-quantized weights or quantize from scratch
    if args.quantized_model_path:
        # Load pre-quantized weights
        # If use_conductance_recovery is True, recover from conductance states
        model, quant_config = load_quantized_model_weights(
            model, 
            args.quantized_model_path, 
            device,
            use_conductance_recovery=args.use_conductance_recovery
        )
        
        # Move model to device with correct dtype
        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        
        # Verify quantization
        verify_quantization(model, bit_width=quant_config['bit_width'], num_layers_to_check=3)
        
        conductance_map = None  # Already saved
    else:
        # Apply M-ANT quantization from scratch and collect conductance mappings
        conductance_map = quantize_model_mant(model, bit_width=args.bit_width, 
                                               group_size=args.group_size,
                                               G_MIN=args.G_min, G_MAX=args.G_max)
        
        # Verify quantization was applied
        verify_quantization(model, bit_width=args.bit_width, num_layers_to_check=3)
        
        # Save quantized model and conductance mappings if requested
        if args.save_quantized:
            save_quantized_model_with_conductance(
                model=model,
                conductance_map=conductance_map,
                save_path=args.save_quantized,
                bit_width=args.bit_width,
                group_size=args.group_size,
                original_model_path=args.model_path,
                G_MIN=args.G_min,
                G_MAX=args.G_max
            )
    
    # Skip evaluation if only quantizing and saving
    if args.skip_evaluation:
        print("\n" + "=" * 80)
        print("EVALUATION SKIPPED")
        print("=" * 80)
        print("Quantized model and conductance mappings have been saved.")
        print(f"Location: {args.save_quantized}")
        print("\nTo evaluate later, run:")
        print(f"  python {os.path.basename(__file__)} \\")
        print(f"    --quantized_model_path {args.save_quantized} \\")
        print(f"    --use_conductance_recovery  # (optional: recover from conductance)")
        print("=" * 80)
    else:
        # Evaluate on PIQA
        print("\n" + "=" * 80)
        print("PIQA BENCHMARK EVALUATION")
        if args.quantized_model_path and args.use_conductance_recovery:
            print("Mode: Weights recovered from conductance states")
        elif args.quantized_model_path:
            print("Mode: Direct quantized weights")
        else:
            print("Mode: Freshly quantized weights")
        print("=" * 80)
        
        accuracy = evaluate_piqa(model, tokenizer, device, num_samples=args.num_samples)
        
        # Determine bit width for display
        if args.quantized_model_path:
            display_bit_width = quant_config['bit_width']
        else:
            display_bit_width = args.bit_width
        
        print(f"\n" + "=" * 80)
        print(f"FINAL PIQA ACCURACY (M-ANT-INT{display_bit_width}): {accuracy:.4f} ({accuracy*100:.2f}%)")
        if args.use_conductance_recovery:
            print(f"Note: Weights were recovered from conductance differential pairs")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-V2-Lite with M-ANT quantization and conductance mapping on PIQA."
    )
    parser.add_argument("--bit_width", type=int, default=4, help="Bit width for quantization.")
    parser.add_argument("--group_size", type=int, default=128, help="Quantization group size.")
    parser.add_argument("--G_min", type=float, default=20.0, help="Minimum conductance in uS.")
    parser.add_argument("--G_max", type=float, default=200.0, help="Maximum conductance in uS.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./DeepSeek-V2-Lite",
        help="Path to the original DeepSeek-V2-Lite checkpoint.",
    )
    parser.add_argument(
        "--quantized_model_path",
        type=str,
        default=None,
        help="Path to a pre-quantized checkpoint directory.",
    )
    parser.add_argument(
        "--save_quantized",
        type=str,
        default="./DeepSeek-V2-Lite-MANT-INT4-G",
        help="Directory used to save quantized weights and conductances.",
    )
    parser.add_argument(
        "--piqa_dir",
        type=str,
        default="./PIQA",
        help="Directory containing PIQA dev.jsonl and dev-labels.lst.",
    )
    parser.add_argument("--num_samples", type=int, default=None, help="Evaluate only the first N PIQA samples.")
    parser.add_argument("--answer_prefix", type=str, default=" ", help="Prefix inserted before each answer choice.")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip PIQA evaluation and only quantize/save.")
    parser.add_argument(
        "--use_conductance_recovery",
        action="store_true",
        help="Recover weights from conductance states when loading a pre-quantized model.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Runtime device: auto/cpu/cuda/cuda:0.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Runtime dtype used while loading the model.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        choices=["none", "auto"],
        help="Optional device_map passed to from_pretrained.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    print("\n" + "=" * 80)
    print(f"DeepSeek-V2-Lite M-ANT-INT{args.bit_width} with Conductance Mapping")
    print("=" * 80)
    print("Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {device}")
    print(f"  DType: {args.dtype}")
    print(f"  Device map: {args.device_map}")
    print(f"  Quantization: INT{args.bit_width} (M-ANT)")
    print(f"  Group size: {args.group_size}")
    print(f"  Conductance range: [{args.G_min}, {args.G_max}] uS")
    print(f"  PIQA dir: {args.piqa_dir}")
    print(f"  Answer prefix: {args.answer_prefix!r}")
    print(f"  Pre-quantized model: {args.quantized_model_path if args.quantized_model_path else 'None (will quantize)'}")
    print(f"  Save quantized: {args.save_quantized if args.save_quantized else 'No'}")
    print(f"  Skip evaluation: {args.skip_evaluation}")
    if not args.skip_evaluation:
        print(f"  Use conductance recovery: {args.use_conductance_recovery}")
    print(f"  Samples: {args.num_samples if args.num_samples else 'All'}")
    print("=" * 80 + "\n")

    model, tokenizer = load_mla_model(
        args.model_path,
        device=device,
        dtype_name=args.dtype,
        device_map=args.device_map,
    )

    if args.quantized_model_path:
        model, quant_config = load_quantized_model_weights(
            model,
            args.quantized_model_path,
            device,
            use_conductance_recovery=args.use_conductance_recovery,
        )
        model.eval()
        verify_quantization(model, bit_width=quant_config["bit_width"], num_layers_to_check=3)
    else:
        conductance_map = quantize_model_mant(
            model,
            bit_width=args.bit_width,
            group_size=args.group_size,
            G_MIN=args.G_min,
            G_MAX=args.G_max,
        )
        verify_quantization(model, bit_width=args.bit_width, num_layers_to_check=3)

        if args.save_quantized:
            save_quantized_model_with_conductance(
                model=model,
                conductance_map=conductance_map,
                save_path=args.save_quantized,
                bit_width=args.bit_width,
                group_size=args.group_size,
                original_model_path=args.model_path,
                G_MIN=args.G_min,
                G_MAX=args.G_max,
            )

    if args.skip_evaluation:
        print("\n" + "=" * 80)
        print("EVALUATION SKIPPED")
        print("=" * 80)
        print("Quantized model and conductance mappings have been saved.")
        if args.save_quantized:
            print(f"Location: {args.save_quantized}")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("PIQA BENCHMARK EVALUATION")
    if args.quantized_model_path and args.use_conductance_recovery:
        print("Mode: Weights recovered from conductance states")
    elif args.quantized_model_path:
        print("Mode: Direct quantized weights")
    else:
        print("Mode: Freshly quantized weights")
    print("=" * 80)

    accuracy = evaluate_piqa(
        model,
        tokenizer,
        device=device,
        num_samples=args.num_samples,
        piqa_dir=args.piqa_dir,
        answer_prefix=args.answer_prefix,
    )

    display_bit_width = quant_config["bit_width"] if args.quantized_model_path else args.bit_width
    print(f"\n" + "=" * 80)
    print(f"FINAL PIQA ACCURACY (M-ANT-INT{display_bit_width}): {accuracy:.4f} ({accuracy * 100:.2f}%)")
    if args.quantized_model_path and args.use_conductance_recovery:
        print("Note: Weights were recovered from conductance differential pairs")
    print("=" * 80)


if __name__ == "__main__":
    main()
