from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from .distribution_output import DistributionOutput
from gluonts.torch.modules.lambda_layer import LambdaLayer

from gluonts.util import lazy_property
from scipy.stats import nbinom


class MixtureArgs(nn.Module):
    def __init__(
        self,
        in_features: int,
        distr_outputs: List[DistributionOutput],
    ) -> None:
        super().__init__()
        self.num_components = len(distr_outputs)
        self.component_projections = nn.ModuleList()
        
        self.proj_mixture_probs = nn.Linear(in_features, self.num_components)
        self.proj_mixture_probs = nn.Softmax(self.proj_mixture_probs)

        for do in distr_outputs:
            self.component_projections.append(
                do.get_args_proj(in_features)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mixture_probs = self.proj_mixture_probs(x)
        component_args = [c_proj(x) for c_proj in self.component_projections]
        return tuple([mixture_probs] + component_args)

    
class MixtureSameFamilyOutput(DistributionOutput):
    @validated()
    def __init__(self, distr_outputs: List[DistributionOutput]) -> None:
        self.num_components = len(distr_outputs)
        self.distr_outputs = distr_outputs

    def get_args_proj(self, in_features: int) -> MixtureArgs:
        return MixtureArgs(self.distr_outputs, in_features)

    # Overwrites the parent class method.
    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        **kwargs,
    ) -> MixtureDistribution:
        mixture_probs = distr_args[0]
        component_args = distr_args[1:]
        return MixtureSameFamily(
            mixture_distribution = Categorical(mixture_probs),
            component_distribution = 
            components=[
                do.distribution(args, loc=loc, scale=scale)
                for do, args in zip(self.distr_outputs, component_args)
            ],
        )

    @property
    def event_shape(self) -> Tuple:
        return self.distr_outputs[0].event_shape

    @property
    def value_in_support(self) -> float:
        return self.distr_outputs[0].value_in_support


    
