# sae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoderDebug(nn.Module):
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

    def forward(self, x, debug: bool = False, use_leaky: bool = False, bypass_shrink: bool = False):
        """
        デバッグ用: debug/use_leaky/bypass_shrink を使って内部確認が可能
        """
        x_centered = x - self.bias  # (N, d_model)
        pre_act = self.encoder(x_centered)  # (N, d_sae)

        if debug:
            try:
                p = pre_act.detach().cpu()
                print(f"[SAE DEBUG] pre_act shape: {p.shape}, min={p.min().item():.6e}, max={p.max().item():.6e}, mean={p.mean().item():.6e}, std={p.std().item():.6e}")
                print(f"[SAE DEBUG] pre_act zero_frac: {(p==0).float().mean().item():.6f}")
            except Exception:
                pass

        # デバッグ挙動：bypass なら pre_act をそのまま、Leaky指定なら LeakyReLU、通常は ReLU
        if bypass_shrink:
            sae_features = pre_act
        else:
            if use_leaky:
                sae_features = F.leaky_relu(pre_act, negative_slope=0.01)
            else:
                sae_features = F.relu(pre_act)

        reconstruction = self.decoder(sae_features) + self.bias
        return reconstruction, sae_features

    def get_loss(self, x):
        reconstruction, sae_features = self.forward(x)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        l1_loss = self.l1_coeff * torch.sum(torch.abs(sae_features)) / sae_features.shape[0]
        return reconstruction_loss, l1_loss