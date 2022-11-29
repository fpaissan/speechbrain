""" Implements different quantizers for model compression.

Authors:
    * Francesco Paissan, 2022
"""
import torch
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import copy
from typing import List
import torch.nn as nn
import speechbrain as sb

def _prepare_model(mod, quant_type):
    model_to_quantize = copy.deepcopy(mod)
    model_to_quantize.eval()
    
    if quant_type == "dynamic":
        qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_dynamic_qconfig)
    elif quant_type == "static":
        qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_qconfig)
    
    return prepare_fx(
        model_to_quantize,
        qconfig_mapping,
        None
    )

def quantize(modules: List[nn.Module], data=None, quant_type="dynamic") -> List[nn.Module]:
    """Quantizes the modules of the brain.
    """
    assert quant_type in ["static", "dynamic"], \
        f"{quant_type} is not supported. Please refer to the documentation."
    
    if data == None and quant_type != "dynamic":
        raise NotImplemented(
            f"{quant_type} is not supported. Refer to the docs for the list of supported quantization types."
        )
    
    if not isinstance(modules, list):
        modules = [modules]
    
    quantized_modules = []

    for m in modules:
        # prepare
        model_prepared = _prepare_model(m, quant_type)
        
        # calibrate activations
        if quant_type == "static":
            model_prepared.eval()
            with torch.no_grad():
                model_prepared(data)
        
        # quantize
        model_quantized = convert_fx(model_prepared)

        quantized_modules.append(model_quantized)

    return quantized_modules