import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        """
        Args:
            input_dim (int): Dimension of the input.
            hidden_dims (list): A list of hidden layer dimensions for the encoder.
            latent_dim (int): Dimension of the latent representation.
        """
        super().__init__()
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (reverse order of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def latentspace(self, x):
        latent = self.encoder(x)
        return latent

    def decode(self, x):
        reconstruction = self.decoder(x)
        return reconstruction


def train_sae(model, training_data, sparsity_weight=0.1):
    # Training loop
    num_epochs = 100
    learning_rate = 1e-3
    batch_size = 16

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        permutation = torch.randperm(training_data.size(0))
        for i in range(0, training_data.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch = training_data[indices]

            optimizer.zero_grad()

            latent = model.latentspace(batch)
            reconstruction = model.decode(latent)

            # Reconstruction loss
            recon_loss = criterion(reconstruction, batch)

            # Sparsity loss (L1 regularization on latent representation)
            sparsity_loss = torch.mean(torch.abs(latent))

            # Combined loss
            loss = recon_loss + sparsity_weight * sparsity_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (training_data.size(0) / batch_size)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")