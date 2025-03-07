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

    def get_sparse_vector(self, x, k=None, threshold=None):
        latent = self.encoder(x)
        
        if k is not None:
            # For each sample, keep only top-k activations
            if latent.dim() > 1:
                values, _ = torch.topk(torch.abs(latent), k=min(k, latent.size(1)), dim=1)
                threshold_per_sample = values[:, -1].unsqueeze(1)
                sparse_latent = torch.where(torch.abs(latent) >= threshold_per_sample, 
                                        latent, 
                                        torch.zeros_like(latent))
            else:
                values, _ = torch.topk(torch.abs(latent), k=min(k, latent.size(0)))
                threshold = values[-1]
                sparse_latent = torch.where(torch.abs(latent) >= threshold,
                                        latent,
                                        torch.zeros_like(latent))
        elif threshold is not None:
            # Keep only activations above threshold
            sparse_latent = torch.where(torch.abs(latent) >= threshold,
                                    latent,
                                    torch.zeros_like(latent))
        else:
            # Default: keep top 10% of activations
            k = max(1, int(0.1 * latent.size(-1)))
            return self.get_sparse_vector(x, k=k)
        
        return sparse_latent

    def decode(self, x):
        reconstruction = self.decoder(x)
        return reconstruction


def train_sae(model, training_data, num_epochs=100, sparsity_weight=0.1, patience=50, lr = 1e-3, batch_size = 16):
    # Training loop

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        permutation = torch.randperm(training_data.size(0))
        for i in range(0, training_data.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch = training_data[indices]

            optimizer.zero_grad()

            # Option 1: Train with regular encoder output
            latent = model.encoder(batch)  # Instead of model.latentspace(batch)
            reconstruction = model.decoder(latent)

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
        losses.append(avg_loss)
        if epoch == 0 or avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}    ",end="\r")

    return losses