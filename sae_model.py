# sae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae, l1_coeff):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model)
        self.bias = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_normal_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.T
        
    def forward(self, x):
        x_centered = x - self.bias
        sae_features = F.relu(self.encoder(x_centered))
        reconstruction = self.decoder(sae_features) + self.bias
        
        return reconstruction, sae_features

    def get_loss(self, x):
        reconstruction, sae_features = self.forward(x)
        
        reconstruction_loss = F.mse_loss(reconstruction, x)
        l1_loss = self.l1_coeff * torch.sum(sae_features) / sae_features.shape[0]
        
        return reconstruction_loss, l1_loss