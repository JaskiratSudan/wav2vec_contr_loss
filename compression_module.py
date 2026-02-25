# compression_module.py


import torch
import torch.nn as nn

class CompressionModule(nn.Module):
    """
    Fuses and processes encoder features to produce a final embedding.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, dropout_rate=0.1):
        """
        Initializes the module layers.

        Args:
            input_dim (int): Feature dimension from encoders.
            hidden_dim (int): Intermediate and final embedding dimension.
            dropout_rate (float): Dropout probability.
        """
        super(CompressionModule, self).__init__()

        # --- Head Module Layers ---
        self.dropout_head = nn.Dropout(p=dropout_rate)
        self.activation_head = nn.LeakyReLU()
        self.mlp3 = nn.Linear(input_dim, hidden_dim)


    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass logic.

        Args:
            encoder_output (torch.Tensor): 4D tensor from an encoder.
                                           Shape: (batch, K, F, T)

        Returns:
            torch.Tensor: Final sequence of shape (batch, hidden_dim, T).
        """
        # 1. Pooling: Average across the layer dimension (K).
        # Input: (B, K, F, T) -> Output: (B, F, T)
        pooled_features = torch.mean(encoder_output, dim=1)
        x = pooled_features

        # --- Apply Head Module ---
        x = self.dropout_head(x)
        x = self.activation_head(x)
        final_sequence = self.mlp3(x.transpose(1, 2)).transpose(1, 2)

        return final_sequence