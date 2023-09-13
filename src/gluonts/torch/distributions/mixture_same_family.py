from typing import Dict, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from .distribution_output import DistributionOutput
from gluonts.torch.modules.lambda_layer import LambdaLayer
from gluonts.core.component import validated

from gluonts.util import lazy_property
from scipy.stats import nbinom
import pdb

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

    
class MixtureArgs(nn.Module):
    def __init__(
        self,
        in_features: int,
        distr_outputs: List[DistributionOutput],
    ) -> None:
        super().__init__()
        self.num_components = len(distr_outputs)
        self.component_projections = nn.ModuleList()
        
        self.proj_mixture_probs = nn.Sequential(
            nn.Linear(in_features, self.num_components),
            LambdaLayer(lambda x: torch.softmax(x, dim=1))
        )        
        
        for do in distr_outputs:
            self.component_projections.append(
                do.get_args_proj(in_features)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mixture_probs = self.proj_mixture_probs(x)
        component_args = [c_proj(x) for c_proj in self.component_projections]
        pdb.set_trace()
        return tuple([mixture_probs] + component_args)


class MixtureSameFamilyOutput(DistributionOutput):
    @validated()
    def __init__(self, distr_outputs: List[DistributionOutput]) -> None:
        self.num_components = len(distr_outputs)
        self.distr_outputs = distr_outputs

    def get_args_proj(self, in_features: int) -> MixtureArgs:
        return MixtureArgs(in_features, self.distr_outputs)

    
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MixtureSameFamily:
        # dealing with single component / broadasted parameters expected by NormalOutput
        mixture_probs = distr_args[0]
        component_args = distr_args[1:]
        nparams = len(component_args[0])
        comp_args_concat = []
        for i in range(nparams):
            comp_args_concat[i] = torch.cat([c[i] for c in component_args])
        
        return MixtureSameFamily(
            mixture_distribution = Categorical(mixture_probs),            
            component_distribution = self.distr_outputs[0].distribution(comp_args_concat, loc=loc, scale=scale)
        )

    @property
    def event_shape(self) -> Tuple:
        return self.distr_outputs[0].event_shape

    @property
    def value_in_support(self) -> float:
        return self.distr_outputs[0].value_in_support


    
# currently dimensions and compatibility not really addressed
