import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.distributions import Categorical, Distribution
from typing import Optional, List

from gluonts.torch.distributions import DistributionOutput

class MixtureDistribution(Distribution):
    def __init__(self, mixture_probs: torch.Tensor, components: List[Distribution]) -> None:
        super().__init__()
        self.mixture_probs = mixture_probs
        self.components = components
        self.categorical = Categorical(probs=mixture_probs)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        mixture_indices = self.categorical.sample(sample_shape)
        component_samples = torch.stack([c.sample(sample_shape) for c in self.components], dim=-1)
        selected_samples = torch.gather(component_samples, -1, mixture_indices.unsqueeze(-1)).squeeze(-1)
        return selected_samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = torch.stack([c.log_prob(x) for c in self.components], dim=-1)
        log_mix_prob = torch.log(self.mixture_probs)
        return torch.logsumexp(log_probs + log_mix_prob, dim=-1)

class MixtureArgs(nn.Module):
    def __init__(self, in_features: int, distr_outputs: List[DistributionOutput]) -> None:
        super().__init__()
        self.num_components = len(distr_outputs)
        self.proj_mixture_probs = nn.Sequential(
            nn.Linear(in_features, self.num_components),
            nn.Softmax(dim=-1)
        )
        self.component_projections = nn.ModuleList([
            do.get_args_proj(in_features) for do in distr_outputs
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mixture_probs = self.proj_mixture_probs(x)
        component_args = [param for proj in self.component_projections for param in proj(x)]
        #if x.shape[0] != 100:
        #    pdb.set_trace()

        return [mixture_probs] + component_args

class MixtureDistributionOutput(DistributionOutput):
    distr_cls: type = MixtureDistribution
    
    def __init__(self, distr_outputs: List[DistributionOutput]) -> None:
        super().__init__()
        self.distr_outputs = distr_outputs

    def get_args_proj(self, in_features: int) -> nn.Module:
        return MixtureArgs(in_features, self.distr_outputs)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MixtureDistribution:
        mixture_probs = distr_args[0]
        component_params_flat = distr_args[1:]
        
        num_params_per_component = len(self.distr_outputs[0].args_dim)
        
        component_args = [tuple(component_params_flat[n:n+num_params_per_component]) 
                          for n in range(0, len(component_params_flat), num_params_per_component)]
        
        components = [
            do.distribution(args, loc=loc, scale=scale)
            for do, args in zip(self.distr_outputs, component_args)
        ]
        return MixtureDistribution(mixture_probs=mixture_probs, components=components)

    @property
    def event_shape(self) -> torch.Size:
        return self.distr_outputs[0].event_shape