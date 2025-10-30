# sae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae, l1_coeff, init_encoder_bias=None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff
  
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model)
        self.bias = nn.Parameter(torch.zeros(d_model))
        nn.init.xavier_normal_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.T

        # 必要なら encoder.bias を小さく初期化して ReLU 系の崩壊を緩和
        if init_encoder_bias is not None:
            self.encoder.bias.data.fill_(float(init_encoder_bias))
 
    def forward(self, x):
        """
        Production forward: デバッグ出力やバイパス引数なしのシンプル実装
        """
        x_centered = x - self.bias  # (N, d_model)
        pre_act = self.encoder(x_centered)  # (N, d_sae)
        # shrink を使わない設定：そのままの encoder 出力を SAE 特徴とする
        sae_features = pre_act
        reconstruction = self.decoder(sae_features) + self.bias
        return reconstruction, sae_features

    def get_loss(self, x):
        reconstruction, sae_features = self.forward(x)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        # L1 正則化は絶対値で計算（符号打ち消しを防ぐ）
        l1_loss = self.l1_coeff * torch.sum(torch.abs(sae_features)) / sae_features.shape[0]
        return reconstruction_loss, l1_loss