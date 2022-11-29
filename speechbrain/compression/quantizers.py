""" Implements different quantizers for model compression.

Authors:
    * Francesco Paissan, 2022
"""
import torch
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import copy

import speechbrain as sb

def _dynamic_quantize(modules):
    if not isinstance(modules, list):
        modules = [modules]
    
    quantized_modules = []

    for m in modules:
        model_to_quantize = copy.deepcopy(m)
        model_to_quantize.eval()

        # a tuple of one or more example inputs are needed to trace the model
        breakpoint()

        # prepare QMapping for dynamic quantization
        qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_dynamic_qconfig)

        # prepare
        model_prepared = prepare_fx(
            model_to_quantize,
            qconfig_mapping,
            None
        )

        breakpoint()          

        # no calibration needed when we only have dynamic/weight_only quantization

        # quantize
        model_quantized = convert_fx(model_prepared)

        quantized_modules.append(model_quantized)

    return modules


def quantize(self, brain: sb.Brain) -> sb.Brain:
    """Quantizes the modules of the brain.

    Returns:
        brain: the brain to be quantized.
    """
    if self.quantization_type == "static":
        return self._static_quantize(brain)
    elif self.quantization_type == "dynamic":
        return self._dynamic_quantize(brain)
    else:
        raise ValueError("Unknown quantization type.")

