# compression_module.py
# Defines the Compression and Classifier Head Module for the SLIM model.

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

        # --- Bottleneck Module Layers ---
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.mlp2 = nn.Linear(hidden_dim, input_dim)

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

        # --- Apply Bottleneck Module ---
        skip_connection = x

        x = self.mlp1(x.transpose(1, 2)).transpose(1, 2)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.mlp2(x.transpose(1, 2)).transpose(1, 2)

        x = x + skip_connection

        # --- Apply Head Module ---
        x = self.dropout_head(x)
        x = self.activation_head(x)
        final_sequence = self.mlp3(x.transpose(1, 2)).transpose(1, 2)

        return final_sequence

# Example usage script.
if __name__ == '__main__':
    # Define dimensions
    BATCH_SIZE = 2
    TIME_STEPS = 249
    FEATURE_DIM = 1024
    STYLE_LAYERS = 11  
    LING_LAYERS = 8    
    HIDDEN_DIM = 256

    # Create dummy 4D feature tensors, simulating encoder outputs.
    dummy_style_output = torch.randn(BATCH_SIZE, STYLE_LAYERS, FEATURE_DIM, TIME_STEPS)
    dummy_ling_output = torch.randn(BATCH_SIZE, LING_LAYERS, FEATURE_DIM, TIME_STEPS)

    print(f"\nInput style encoder output shape: {dummy_style_output.shape}")
    print(f"Input linguistic encoder output shape: {dummy_ling_output.shape}")

    print("\nInitializing CompressionModule...")
    try:
        compression_module = CompressionModule(
            input_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout_rate=0.1
        )
        compression_module.eval()
        print("CompressionModule initialized successfully.")

        print("\nPerforming forward pass...")
        with torch.no_grad():
            style_dependency_features = compression_module(dummy_style_output)
            ling_dependency_features = compression_module(dummy_ling_output)
        
        print("Forward pass successful.")
        print(f"\nStyle Dependency Features Shape: {style_dependency_features.shape}")
        print(f"Linguistics Dependency Features Shape: {ling_dependency_features.shape}\n")


    except Exception as e:
        print(f"\nAn error occurred during the example usage: {e}")

