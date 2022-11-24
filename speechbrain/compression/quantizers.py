""" Implements different quantizers for model compression.

Authors:
    * Francesco Paissan, 2022
"""
import torch
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import copy

import speechbrain as sb

backend = "fbgemm" # or "qnnpack" for ARM archs

class PostTrainingQuantization():
    def __init__(
        self,
        quantization_type: str,
        modules_to_quantize: list[str],
        input_shapes: dict,
        qconfig_dict: dict = {"": torch.quantization.get_default_qconfig(backend)},
        isInterface: bool = True,
    ) -> None:
        self.quantization_type = quantization_type
        self.qconfig_dict = qconfig_dict
        self.modules_to_quantize = modules_to_quantize
        self.isInterface = isInterface
        self.input_shapes = input_shapes

    def _static_quantize(self, brain):
        pass

    def _dynamic_quantize(self, brain):
        if not self.isInterface:
            if self.modules_to_quantize == "all":
                self.modules_to_quantize = list(brain.modules.keys())

            for module_name in self.modules_to_quantize:
                if "norm" in module_name:
                    continue
                
                model_to_quantize = copy.deepcopy(brain.modules[module_name])
                model_to_quantize.eval()
                
                dummy = torch.randn(self.input_shapes[module_name])
                
                # a tuple of one or more example inputs are needed to trace the model
                breakpoint()
                # prepare
                model_prepared = prepare_fx(
                    model_to_quantize,
                    self.qconfig_dict,
                    dummy
                )

                breakpoint()                
                # no calibration needed when we only have dynamic/weight_only quantization
                # quantize
                model_quantized = convert_fx(model_prepared)

                brain.modules[module_name] = model_quantized

        return brain


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

