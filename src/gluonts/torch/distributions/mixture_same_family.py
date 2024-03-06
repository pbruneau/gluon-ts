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
        num_components: int,
        distr_output: DistributionOutput,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_components = num_components
        self.component_projections = nn.ModuleList()
        
        self.proj_mixture_probs = nn.Sequential(
            nn.Linear(in_features, self.num_components),
            LambdaLayer(lambda x: torch.softmax(x, dim=2))
        )
        
        # hack to have single tuple of linear projections with num_component output dims
        # to comply with mytorch way to handle mixtures
        distr_output.args_dim = {key: num_components for key in distr_output.args_dim.keys()}
        self.component_projection = distr_output.get_args_proj(in_features)
        
        self.nparams = len(self.component_projection.args_dim.keys())
        #pdb.set_trace()
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # input has shape (batch_size, nsteps, nhidden)
        # proj_mixture_probs: (nhidden, ncomp)
        # component_projections: list[ncomp] of list[nparams] (tuples?) (nhidden, 1)

        mixture_probs = self.proj_mixture_probs(x)
        component_proj = self.component_projection(x)

        #if x.shape[0] != 100:
        #    pdb.set_trace()
        #pdb.set_trace()
        
        # mixture probs: (batch_size, nsteps, ncomp)
        # component_args: list[ncomp*nparams] (batch_size, nsteps) (ncomp major)
        # does not play nice with nested list as with MXNet
        
        return tuple([mixture_probs] + list(component_proj))


class MixtureSameFamilyOutput(DistributionOutput):
    @validated()
    def __init__(self, 
                 # using the torch way of representing mixtures
                 # with mixture_same_family
                 distr_output: DistributionOutput, 
                 num_components: int) -> None:
        self.num_components = num_components
        self.distr_output = distr_output
        self.nparams = None

    def get_args_proj(self, in_features: int) -> MixtureArgs:
        mix_args = MixtureArgs(in_features, self.num_components, self.distr_output)
        self.nparams = mix_args.nparams
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


        ## mimic device used elsewhere
        #device = scale.device
        #
        #comp_args_concat = []
        #for j in range(self.nparams):
        #    #if i==0 and j==0:
        #    #    pdb.set_trace()
        #    tensor = torch.empty((batch_size, nsteps, self.num_components), device=device)
        #    
        #    for i in range(self.num_components):
        #        index = i*self.nparams + j
        #        tensor[:, :, i] = component_args[index]
        #    comp_args_concat.append(tensor)
        #
        ##pdb.set_trace()
        
        return MixtureSameFamily(
            mixture_distribution = Categorical(mixture_probs),
            # unsqueeze so that scale matches the mixture specification
            component_distribution = self.distr_output.distribution(component_args, loc=loc, \
                                                                        scale=scale.unsqueeze(-1))
        )

    @property
    def event_shape(self) -> Tuple:
        return self.distr_output.event_shape

    @property
    def value_in_support(self) -> float:
        return self.distr_output.value_in_support


    