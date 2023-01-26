import numpy as np


import torch
import torch.nn as nn


class FLow_harmonizer(nn.Module):

    def __init__(self, flows, harmonizer_net):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.flows = nn.ModuleList(flows.flows)
        self.harmonizer = harmonizer_net
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, imgs, masks):
        z = self.harmonizer.forward(imgs)
        z = torch.clamp(z, min=0, max=255)
        z[masks==0] = 0
        ldj = torch.zeros(z.shape[0]).cuda()
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(z, ldj)
    
    def forward_base(self, z):
        ldj = torch.zeros(z.shape[0]).cuda()
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(z, ldj)

    def encode(self, z, ldj):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z_splits = []
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj, z_splits

    def _get_likelihood(self, z, ldj, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj, z_splits = self.encode(z, ldj)
        log_pz = self.prior.log_prob(z).sum(dim=[1,2,3])
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(z.shape[1:])
        return bpd.mean() if not return_ll else log_px