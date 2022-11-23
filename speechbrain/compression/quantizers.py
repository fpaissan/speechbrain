""" Implements different quantizers for model compression.

Authors:
    * Francesco Paissan, 2022
"""
import torch
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.quantization.quantize_fx as quantize_fx
import copy

import speechbrain as sb

backend = "fbgemm" # or "qnnpack" for ARM archs

class PostTrainingQuantization():
    def __init__(
        self,
        quantization_type: str,
        modules_to_quantize: list[str],
        qconfig_dict: dict = {"": torch.quantization.get_default_qconfig(backend)},
        isInterface: bool = True,
    ) -> None:
        self.quantization_type = quantization_type
        self.qconfig_dict = qconfig_dict
        self.modules_to_quantize = modules_to_quantize
        self.isInterface = isInterface

    def _static_quantize(self, brain):
        pass

    def _dynamic_quantize(self, brain):
        if self.isInterface:
            if self.modules_to_quantize == "all":
                self.modules_to_quantize = list(brain.mods.keys())

            for module_name in self.modules_to_quantize:
                model_to_quantize = copy.deepcopy(brain.mods[module_name])
                model_to_quantize.eval()

                # a tuple of one or more example inputs are needed to trace the model

                # prepare
                model_prepared = quantize_fx.prepare_fx(
                    model_to_quantize,
                    self.qconfig_dict,
                    None
                )

                # no calibration needed when we only have dynamic/weight_only quantization
                # quantize
                model_quantized = quantize_fx.convert_fx(model_prepared)

                import pdb; pdb.set_trace()

                brain.mods[module_name] = model_quantized

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

