from typing import Dict, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
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
        self.in_features = in_features
        self.num_components = len(distr_outputs)
        self.component_projections = nn.ModuleList()
        
        self.proj_mixture_probs = nn.Sequential(
            nn.Linear(in_features, self.num_components),
            LambdaLayer(lambda x: torch.softmax(x, dim=2))
        )
        
        for do in distr_outputs:
            self.component_projections.append(
                do.get_args_proj(in_features)
            )
        self.per_comp_len = len(self.component_projections[0].args_dim.keys())
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mixture_probs = self.proj_mixture_probs(x)
        # input has shape (batch_size, nsteps, nhidden)
        # proj_mixture_probs: (nhidden, ncomp)
        # component_projections: list[ncomp] of list[nparams] (tuples?) (nhidden, 1)

        # flattening component hierarchy
        component_args = []
        for c_proj in self.component_projections:
            c_proj = c_proj(x)
            for i in range(len(c_proj)):
                component_args.append(c_proj[i])

        #if x.shape[0] != 100:
        #    pdb.set_trace()
        
        # mixture probs: (batch_size, nsteps, ncomp)
        # component_args: list[ncomp*nparams] (batch_size, nsteps) (ncomp major)
        # does not play nice with nested list as with MXNet
        
        return tuple([mixture_probs] + component_args)


class MixtureSameFamilyOutput(DistributionOutput):
    @validated()
    def __init__(self, distr_outputs: List[DistributionOutput]) -> None:
        self.num_components = len(distr_outputs)
        self.distr_outputs = distr_outputs
        self.per_comp_len = None

    def get_args_proj(self, in_features: int) -> MixtureArgs:
        mix_args = MixtureArgs(in_features, self.distr_outputs)
        self.per_comp_len = mix_args.per_comp_len
        return mix_args

    
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MixtureSameFamily:
        # dealing with single component / broadasted parameters expected by NormalOutput
        # and flattened component structure
        mixture_probs = distr_args[0]
        component_args = distr_args[1:]
        batch_size = distr_args[0].shape[0]
        nsteps = distr_args[0].shape[1]
        
        #pdb.set_trace()
        
        # mixture_distribution expects (batch_size, nsteps, ncomp)
        # mixture_probs is (batch_size, nsteps, ncomp)
        
        # component_distribution argument of MixtureSameFamily expects list[nparams] (batch_size, nsteps, ncomp)
        # component_args is list[ncomp*nparams] (batch_size, nsteps) (as obtained from forward)
        
        # in MXNet, mixture_probs is (batch_size, nsteps, ncomp) 
        # component_args has cells (batch_size, nstep) at this stage
        # loc is None
        # scale is (batch_size, 1)
        
        # with NormalOutput, list[nparams] (batch_size, nsteps), passed to base distribution constructor (which expands it),
        # so output is consistent
        # loc is None
        # shape is (batch_size, 1)
        
        # for consistency with single output and class signature, argments to MixtureSameFamily should be:
        # mixture_distribution: (batch_size, nsteps, ncomp)
        # component_distribution: list[nparams] (batch_size, nsteps, ncomp)

        # mimic device used elsewhere
        device = scale.device
        
        comp_args_concat = []
        for j in range(self.per_comp_len):
            #if i==0 and j==0:
            #    pdb.set_trace()
            tensor = torch.empty((batch_size, nsteps, self.num_components), device=device)
            
            for i in range(self.num_components):
                index = i*self.per_comp_len + j
                tensor[:, :, i] = component_args[index]
            comp_args_concat.append(tensor)
        
        #pdb.set_trace()
        
        return MixtureSameFamily(
            mixture_distribution = Categorical(mixture_probs),
            # unsqueeze so that scale matches the mixture specification
            component_distribution = self.distr_outputs[0].distribution(comp_args_concat, loc=loc, \
                                                                        scale=scale.unsqueeze(-1))
        )

    @property
    def event_shape(self) -> Tuple:
        return self.distr_outputs[0].event_shape

    @property
    def value_in_support(self) -> float:
        return self.distr_outputs[0].value_in_support


    