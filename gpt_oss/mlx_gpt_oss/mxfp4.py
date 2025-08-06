"""
MXFP4 (Microscaling FP4) implementation for GPT-OSS weights.

MXFP4 uses E2M1 format (2-bit exponent, 1-bit mantissa) with block-wise scaling
to achieve 4.25 bits per parameter effective encoding.

Based on OCP Microscaling Formats (MX) Specification v1.0
"""

import mlx.core as mx
import numpy as np
from typing import Tuple, Union


class MXFP4Codec:
    """MXFP4 encoder/decoder implementation."""
    
    # MXFP4 E2M1 format constants
    EXPONENT_BITS = 2
    MANTISSA_BITS = 1
    EXPONENT_BIAS = 1
    BLOCK_SIZE = 32  # Standard block size for MX formats
    
    # E2M1 lookup table for dequantization
    # Format: [sign][exponent][mantissa] -> float value
    E2M1_VALUES = {
        # Positive values
        0b000: 0.0,      # +0
        0b001: 0.5,      # +0.5 * 2^(-1)
        0b010: 1.0,      # +1.0 * 2^0
        0b011: 1.5,      # +1.5 * 2^0
        0b100: 2.0,      # +1.0 * 2^1
        0b101: 3.0,      # +1.5 * 2^1
        0b110: 4.0,      # +1.0 * 2^2
        0b111: 6.0,      # +1.5 * 2^2
        # Negative values (with sign bit)
        0b1000: -0.0,    # -0
        0b1001: -0.5,    # -0.5 * 2^(-1)
        0b1010: -1.0,    # -1.0 * 2^0
        0b1011: -1.5,    # -1.5 * 2^0
        0b1100: -2.0,    # -1.0 * 2^1
        0b1101: -3.0,    # -1.5 * 2^1
        0b1110: -4.0,    # -1.0 * 2^2
        0b1111: -6.0,    # -1.5 * 2^2
    }
    
    @classmethod
    def pack_mxfp4_block(cls, values: np.ndarray, block_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pack float32 values into MXFP4 format.
        
        Returns:
            packed_data: uint8 array with packed 4-bit values
            scales: int8 array with block-wise scales
        """
        if block_size is None:
            block_size = cls.BLOCK_SIZE
            
        # Reshape into blocks
        if values.size % block_size != 0:
            # Pad to block boundary
            pad_size = block_size - (values.size % block_size)
            values = np.pad(values, (0, pad_size), mode='constant', constant_values=0)
        
        values_blocked = values.reshape(-1, block_size)
        num_blocks = values_blocked.shape[0]
        
        # Compute block-wise scales
        scales = []
        packed_blocks = []
        
        for block in values_blocked:
            # Find the maximum absolute value in the block
            max_abs = np.max(np.abs(block))
            
            if max_abs == 0:
                scale_exp = -127  # Minimum scale for zero block
            else:
                # Compute scale to fit values in E2M1 range
                # E2M1 max absolute value is 6.0
                scale_exp = int(np.floor(np.log2(max_abs / 6.0))) if max_abs > 0 else -127
                scale_exp = max(-127, min(127, scale_exp))  # Clamp to int8 range
            
            scales.append(scale_exp)
            
            # Scale the block
            scale_factor = 2.0 ** scale_exp
            scaled_block = block / scale_factor
            
            # Quantize to MXFP4
            quantized_block = cls._quantize_to_e2m1(scaled_block)
            packed_blocks.append(quantized_block)
        
        # Pack 4-bit values into uint8 array
        packed_data = np.zeros((num_blocks, block_size // 2), dtype=np.uint8)
        
        for i, block in enumerate(packed_blocks):
            for j in range(0, block_size, 2):
                # Pack two 4-bit values into one uint8
                val1 = int(block[j]) & 0xF
                val2 = int(block[j + 1]) & 0xF if j + 1 < block_size else 0
                packed_data[i, j // 2] = (val1 << 4) | val2
        
        return packed_data.flatten(), np.array(scales, dtype=np.int8)
    
    @classmethod
    def unpack_mxfp4_block(cls, packed_data: np.ndarray, scales: np.ndarray, 
                          original_shape: Tuple[int, ...], block_size: int = None) -> np.ndarray:
        """
        Unpack MXFP4 data back to float32.
        
        Args:
            packed_data: uint8 array with packed 4-bit values
            scales: int8 array with block-wise scales  
            original_shape: Original tensor shape
            block_size: Block size used for packing
            
        Returns:
            Unpacked float32 array
        """
        if block_size is None:
            block_size = cls.BLOCK_SIZE
            
        total_elements = np.prod(original_shape)
        num_blocks = len(scales)
        
        # Unpack 4-bit values
        unpacked_values = []
        
        packed_data = packed_data.reshape(num_blocks, block_size // 2)
        
        for block_idx in range(num_blocks):
            scale_exp = scales[block_idx]
            scale_factor = 2.0 ** scale_exp
            
            block_values = []
            for byte_idx in range(block_size // 2):
                packed_byte = packed_data[block_idx, byte_idx]
                
                # Extract two 4-bit values
                val1 = (packed_byte >> 4) & 0xF
                val2 = packed_byte & 0xF
                
                # Dequantize from E2M1
                float_val1 = cls._dequantize_from_e2m1(val1) * scale_factor
                float_val2 = cls._dequantize_from_e2m1(val2) * scale_factor
                
                block_values.extend([float_val1, float_val2])
            
            unpacked_values.extend(block_values[:block_size])
        
        # Trim to original size and reshape
        result = np.array(unpacked_values[:total_elements], dtype=np.float32)
        return result.reshape(original_shape)
    
    @classmethod
    def _quantize_to_e2m1(cls, values: np.ndarray) -> np.ndarray:
        """Quantize float values to 4-bit E2M1 format."""
        quantized = np.zeros(values.shape, dtype=np.uint8)
        
        for i, val in enumerate(values.flat):
            # Find closest E2M1 value
            min_diff = float('inf')
            best_code = 0
            
            for code, e2m1_val in cls.E2M1_VALUES.items():
                diff = abs(val - e2m1_val)
                if diff < min_diff:
                    min_diff = diff
                    best_code = code
            
            quantized.flat[i] = best_code
        
        return quantized
    
    @classmethod
    def _dequantize_from_e2m1(cls, code: int) -> float:
        """Dequantize 4-bit E2M1 code to float value."""
        return cls.E2M1_VALUES.get(code & 0xF, 0.0)


def convert_to_mxfp4(tensor: mx.array, block_size: int = 32) -> Tuple[mx.array, mx.array]:
    """
    Convert MLX tensor to MXFP4 format.
    
    Args:
        tensor: Input MLX tensor
        block_size: Block size for microscaling
        
    Returns:
        packed_data: Packed MXFP4 data as uint8 array
        scales: Block-wise scales as int8 array
    """
    # Convert to numpy for processing
    np_tensor = np.array(tensor).astype(np.float32)
    original_shape = np_tensor.shape
    
    # Pack to MXFP4
    packed_data, scales = MXFP4Codec.pack_mxfp4_block(np_tensor.flatten(), block_size)
    
    return mx.array(packed_data), mx.array(scales)


def convert_from_mxfp4(packed_data: mx.array, scales: mx.array, 
                      original_shape: Tuple[int, ...], block_size: int = 32) -> mx.array:
    """
    Convert MXFP4 format back to MLX tensor.
    
    Args:
        packed_data: Packed MXFP4 data as uint8 array
        scales: Block-wise scales as int8 array
        original_shape: Original tensor shape
        block_size: Block size used for packing
        
    Returns:
        Reconstructed MLX tensor
    """
    # Convert to numpy for processing
    np_packed = np.array(packed_data, dtype=np.uint8)
    np_scales = np.array(scales, dtype=np.int8)
    
    # Unpack from MXFP4
    unpacked = MXFP4Codec.unpack_mxfp4_block(np_packed, np_scales, original_shape, block_size)
    
    return mx.array(unpacked)


def is_mxfp4_tensor(tensor_dict: dict, key: str) -> bool:
    """
    Check if a tensor is stored in MXFP4 format.
    
    MXFP4 tensors in GPT-OSS are identified by:
    1. Being MoE weights (90%+ of parameters)
    2. Having corresponding scale tensors
    """
    # Check if this is an MoE weight
    moe_keywords = ['experts', 'gate_proj', 'up_proj', 'down_proj']
    is_moe_weight = any(keyword in key for keyword in moe_keywords)
    
    if not is_moe_weight:
        return False
    
    # Check for corresponding scale tensor
    scale_key = key + '_scales'
    has_scales = scale_key in tensor_dict
    
    # Check if tensor has typical MXFP4 characteristics
    tensor = tensor_dict[key]
    is_uint8 = hasattr(tensor, 'dtype') and str(tensor.dtype) == 'uint8'
    
    return has_scales and is_uint8


def load_mxfp4_weights(weight_dict: dict) -> dict:
    """
    Load weights with proper MXFP4 handling.
    
    Args:
        weight_dict: Dictionary of weight tensors from safetensors
        
    Returns:
        Dictionary with MXFP4 weights converted to float32
    """
    converted_weights = {}
    processed_keys = set()
    
    for key, tensor in weight_dict.items():
        if key in processed_keys:
            continue
            
        if is_mxfp4_tensor(weight_dict, key):
            # Handle MXFP4 tensor
            scale_key = key + '_scales'
            
            if scale_key in weight_dict:
                print(f"Converting MXFP4 tensor: {key}")
                
                packed_data = mx.array(tensor)
                scales = mx.array(weight_dict[scale_key])
                
                # We need the original shape - try to infer from tensor name
                # In practice, this would come from model metadata
                original_shape = packed_data.shape  # Placeholder
                
                # Convert from MXFP4
                converted_tensor = convert_from_mxfp4(packed_data, scales, original_shape)
                converted_weights[key] = converted_tensor
                
                processed_keys.add(key)
                processed_keys.add(scale_key)
            else:
                print(f"Warning: No scale tensor found for MXFP4 weight {key}")
                converted_weights[key] = mx.array(tensor)
        else:
            # Regular tensor
            converted_weights[key] = mx.array(tensor)
            processed_keys.add(key)
    
    return converted_weights