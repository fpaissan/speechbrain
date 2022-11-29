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

def _dynamic_quantize(modules):
    if not isinstance(modules, list):
        modules = [modules]
    
    quantized_modules = []

    for m in modules:
        model_to_quantize = copy.deepcopy(m)
        model_to_quantize.eval()

        # prepare QMapping for dynamic quantization
        qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_dynamic_qconfig)

        # prepare
        model_prepared = prepare_fx(
            model_to_quantize,
            qconfig_mapping,
            None
        )

        # no calibration needed when we only have dynamic/weight_only quantization

        # quantize
        model_quantized = convert_fx(model_prepared)

        quantized_modules.append(model_quantized)

    return quantized_modules


def quantize(modules: List[nn.Module], quant_type="dynamic") -> List[nn.Module]:
    """Quantizes the modules of the brain.
    """
#    if self.quantization_type == "static":
#        return self._static_quantize(brain)
    if quant_type == "dynamic":
        return _dynamic_quantize(modules)
    else:
        raise NotImplemented(f"Unknown quantization type {quant_type}.")

