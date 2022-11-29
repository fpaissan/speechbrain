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

def _static_quantize(modules, data):
    assert data != None, "Can't quantize without calibration data for Post Training static quantization."
    if not isinstance(modules, list):
        modules = [modules]
    
    quantized_modules = []

    for m in modules:
        # prepare
        model_prepared = _prepare_model(m, "static")
        
        # calibrate activations
        model_prepared.eval()
        with torch.no_grad():
            model_prepared(data)
        
        # quantize
        model_quantized = convert_fx(model_prepared)

        quantized_modules.append(model_quantized)

    return quantized_modules

def _dynamic_quantize(modules):
    if not isinstance(modules, list):
        modules = [modules]
    
    quantized_modules = []

    for m in modules:
        # prepare
        model_prepared = _prepare_model(m, "dynamic")
        
        # quantize
        model_quantized = convert_fx(model_prepared)

        quantized_modules.append(model_quantized)

    return quantized_modules


def quantize(modules: List[nn.Module], data=None, quant_type="dynamic") -> List[nn.Module]:
    """Quantizes the modules of the brain.
    """
    
    if quant_type == "dynamic":
        return _dynamic_quantize(modules)
    elif quant_type == "static":
        return _static_quantize(modules, data)
    else:
        raise NotImplemented(f"Unknown quantization type {quant_type}.")

