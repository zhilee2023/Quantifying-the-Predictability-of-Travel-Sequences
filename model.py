import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


############################################
# Utility: Sequence Padding for 1D Convolution
############################################
def padded_sequence(
    sequence: torch.Tensor,
    kernel_size: int,
    padding_mode: str = "cycle",
) -> torch.Tensor:
    """
    Pads a time-series tensor along the temporal dimension for 1D convolutions.

    Args:
        sequence (torch.Tensor): Input tensor of shape (B, T, C).
        kernel_size (int): Size of the convolutional kernel.
        padding_mode (str): "zero" or "cycle". Defaults to "cycle".

    Returns:
        torch.Tensor: Padded tensor of shape (B, T + kernel_size - 1, C).
    """
    B, T, C = sequence.shape
    pad_len = kernel_size - 1

    if padding_mode == "zero":
        pad = torch.zeros((B, pad_len, C), device=sequence.device, dtype=sequence.dtype)
        return torch.cat([sequence, pad], dim=1)

    if padding_mode == "cycle":
        pad = sequence[:, :pad_len, :].clone()
        return torch.cat([sequence, pad], dim=1)

    raise ValueError("padding_mode must be 'zero' or 'cycle'.")


############################################
# Positional Encoding for Transformer
############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if d_model % 2 != 0:
            # If d_model is odd, fix the last column
            pe[:, -1] = torch.sin(position * div_term[-1]).squeeze()

        self.register_buffer("pe", pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (B, T, D).

        Returns:
            torch.Tensor: (B, T, D) after adding positional encoding.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


############################################
# 1D Convolutional Encoder/Decoder Block
############################################
class EncoderDecoder1D(nn.Module):
    """
    Stacked 1D convolutional block for encoding or decoding.

    For the encoder:
        - in_channels = original feature dimension
        - out_channels = embedding dimension

    For the decoder:
        - in_channels = embedding dimension
        - out_channels = original feature dimension
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        time_steps: int,
        num_layers: int,
        padding_mode: str = "cycle",
    ):
        super().__init__()
        layers = []
        curr_in = in_channels

        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(
                in_channels=curr_in,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                padding_mode="zeros",
            ))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            curr_in = hidden_channels

        # Final layer maps to out_channels
        layers.append(nn.Conv1d(
            in_channels=curr_in,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            padding_mode="zeros",
        ))

        self.net = nn.Sequential(*layers)
        self.kernel_size = kernel_size
        self.time_steps = time_steps
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (B, T, C_in).

        Returns:
            torch.Tensor: Shape (B, T, C_out).
        """
        # Pad along time dimension and then apply Conv1d expects (B, C, T)
        x = padded_sequence(x, self.kernel_size, self.padding_mode)
        x = x.permute(0, 2, 1).contiguous()  # (B, C_in, T + pad)
        out = self.net(x)  # (B, C_out, T + pad)
        out = out[:, :, : self.time_steps]  # truncate to original T
        return out.permute(0, 2, 1).contiguous()  # (B, T, C_out)


############################################
# Vector Quantizer (VQ-VAE)
############################################
class VectorQuantizer(nn.Module):
    """
    VQ-VAE quantization layer.

    Args:
        num_embeddings (int): Number of discrete codebook vectors.
        embedding_dim (int): Dimension of each codebook vector.
        commitment_cost (float): Weight for commitment loss.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook embedding
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(
            self.embeddings.weight,
            -1.0 / num_embeddings,
            1.0 / num_embeddings,
        )

    def forward(
        self,
        z: torch.Tensor,  # shape (B, T, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantizes the input z.

        Args:
            z (torch.Tensor): (B, T, D)

        Returns:
            quantized (torch.Tensor): (B, T, D) – straight-through quantized output
            vq_loss (torch.Tensor): scalar VQ loss
            encoding_indices (torch.LongTensor): (B, T) discrete indices
            avg_soft_probs (torch.Tensor): (num_embeddings,) average soft assignments
        """
        B, T, D = z.shape
        flat_z = z.view(-1, D)  # (B*T, D)

        # Compute L2 distances (B*T, num_embeddings)
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embeddings.weight.t())
        )

        # Soft assignments (for entropy calculation)
        temperature = 0.05
        soft_probs = F.softmax(-distances / temperature, dim=1).view(B, T, self.num_embeddings)
        avg_soft_probs = soft_probs.mean(dim=(0, 1))  # (num_embeddings,)

        # Hard assignments
        encoding_indices = torch.argmin(distances, dim=1).view(B, T)  # (B, T)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # (B, T, num_embeddings)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view(B, T, D)

        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        return quantized, vq_loss, encoding_indices, avg_soft_probs


############################################
# Prediction Module (Transformer-based)
############################################
class PredModulePhi(nn.Module):
    """
    Predicts next codebook index from past one-hot embeddings using a TransformerEncoder.

    Args:
        input_dim (int): Dimension of input features (e.g., one-hot or embedding).
        hidden_dim (int): Dimension inside Transformer.
        output_dim (int): Number of codebook classes to predict.
        time_steps (int): Sequence length T for mask construction.
        nhead (int): Number of attention heads.
        num_layers (int): Number of TransformerEncoder layers.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        time_steps: int,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_steps = time_steps

        # Input projection
        self.fc_in = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        # Causal mask for autoregressive prediction
        mask = torch.triu(torch.ones((time_steps, time_steps), dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to codebook logits
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, T, input_dim) one-hot or embedding input.

        Returns:
            torch.Tensor: (B, T, output_dim) logits for next-class prediction.
        """
        # Project to hidden_dim
        x = self.fc_in(x)  # (B, T, hidden_dim)
        x = self.pos_encoder(x)  # add positional encodings

        # Apply causal padding if needed (not strictly required if no conv)
        # x = padded_sequence(x, kernel_size=1, padding_mode="zero")  # optional

        # Transformer with causal mask
        out = self.transformer(x, mask=self.mask)  # (B, T, hidden_dim)

        # Project to logits
        logits = self.fc_out(out)  # (B, T, output_dim)
        return logits


############################################
# EC-VQVAE Model: Encoder + VQ + Decoder + Predictor
############################################
class EC_VQVAE(nn.Module):
    """
    Residual VQ-VAE with optional prediction head and entropy regularization.

    Args:
        in_channels (int): Input feature dimension (C).
        hidden_channels (int): Width of intermediate conv layers.
        codebook_size (int): Number of discrete embeddings.
        embedding_dim (int): Dimension of each codebook vector.
        commitment_cost (float): Weight for the VQ commitment loss.
        time_steps (int): Temporal length T for input/output.
        num_conv_layers (int): Number of 1D conv layers in encoder/decoder.
        kernel_size (int): Conv kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float,
        time_steps: int,
        num_conv_layers: int,
        kernel_size: int,
    ):
        super().__init__()
        self.time_steps = time_steps

        # Encoder: (B, T, C) -> (B, T, embedding_dim)
        self.encoder = EncoderDecoder1D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            time_steps=time_steps,
            num_layers=num_conv_layers,
            padding_mode="cycle",
        )

        # Vector Quantizer
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder: (B, T, embedding_dim) -> (B, T, C)
        self.decoder = EncoderDecoder1D(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            time_steps=time_steps,
            num_layers=num_conv_layers,
            padding_mode="cycle",
        )

        # Prediction head: from reconstructions to codebook logits
        self.predictor = EncoderDecoder1D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=codebook_size,
            kernel_size=kernel_size,
            time_steps=time_steps,
            num_layers=num_conv_layers,
            padding_mode="cycle",
        )

        # PredModulePhi(
        #     input_dim=in_channels,
        #     hidden_dim=hidden_channels,
        #     output_dim=codebook_size,
        #     time_steps=time_steps,
        #     nhead=4,
        #     num_layers=3,
        #     dropout=0.1,
        # )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input batch (B, T, C).

        Returns:
            x_recon (torch.Tensor): Reconstructed (B, T, C).
            vq_loss (torch.Tensor): Scalar VQ loss.
            entropy_loss (torch.Tensor): Scalar entropy of soft assignments.
            pred_loss (torch.Tensor): Scalar cross-entropy prediction loss.
            encoding_indices (torch.LongTensor): (B, T) discrete codes.
        """
        # Encode
        z = self.encoder(x)  # (B, T, D)

        # Quantize
        quantized, vq_loss, encoding_indices, avg_soft_probs = self.vq(z)
        entropy_loss = - (avg_soft_probs * avg_soft_probs.clamp(min=1e-10).log()).sum()

        # Decode
        x_recon = self.decoder(quantized)

        # Prediction: detach recon to avoid gradients flowing into decoder
        with torch.no_grad():
            quantized_detached = quantized.detach()
        logits = self.predictor(self.decoder(quantized_detached))  # (B, T, codebook_size)

        # Compute prediction loss against ground-truth encoding_indices
        B, T = encoding_indices.shape
        logits_flat = logits.view(B * T, -1)
        idx_flat = encoding_indices.view(-1)
        pred_loss = F.cross_entropy(logits_flat, idx_flat)

        return x_recon, vq_loss, entropy_loss, pred_loss, encoding_indices


############################################
# Training Loop
############################################
def model_train(
    model: EC_VQVAE,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int,
    output_file: str,
    pretrain_epochs: int = 1,
    step_size: int = 2,
    sigma: float = 1.1,
    gamma: float = 0.2,
    D_target: float = 2.5,
    lr: float = 1e-3,
) -> EC_VQVAE:
    """
    Trains the EC_VQVAE model with optional augmented Lagrangian after pretraining.

    Args:
        model (EC_VQVAE): The model to train.
        dataloader (DataLoader): Yields x of shape (B, T, C).
        device (torch.device): "cpu" or "cuda".
        num_epochs (int): Total number of epochs.
        output_file (str): File path to log losses.
        pretrain_epochs (int): Number of epochs to train without ALM.
        step_size (int): Step size for LR scheduler.
        sigma (float): Multiplier for rho update.
        gamma (float): Factor for LR scheduler decay.
        D_target (float): Target reconstruction loss threshold for inequality constraint.
        lr (float): Initial learning rate.

    Returns:
        EC_VQVAE: The trained model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Initialize ALM multipliers
    rho = 1.0
    lambda_eq = 0.5
    mu_ineq = 0.5

    for epoch in range(num_epochs+pretrain_epochs):
        model.train()
        running_recon = 0.0
        running_vq = 0.0
        running_pred = 0.0
        running_entropy = 0.0
        batches = 0

        for x_batch in dataloader:
            x_batch = x_batch.float().to(device)  # (B, T, C)
            x_recon, vq_loss, entropy_loss, pred_loss, _ = model(x_batch)

            # Reconstruction loss (MSE scaled by feature dimension)
            recon_loss = F.mse_loss(x_recon, x_batch) * x_batch.size(-1)

            # Augmented Lagrangian after pretrain
            if epoch >= pretrain_epochs:
                # Equality constraint: pred_loss == 0 ideally
                h_val = pred_loss
                # Inequality constraint: recon_loss <= D_target
                g_val = recon_loss - D_target

                total_loss = (
                    entropy_loss
                    + lambda_eq * h_val
                    + 0.5 * rho * (h_val**2)
                    + mu_ineq * (F.relu(g_val)+vq_loss)
                    + 0.5 * rho * ((F.relu(g_val)+vq_loss) ** 2)
                )

                # Update multipliers
                lambda_eq = lambda_eq + rho * h_val.detach()
                mu_ineq = torch.clamp(mu_ineq + rho * g_val.detach(), min=0.0)

                # Possibly increase rho if constraint violation is too large
                if (recon_loss.item() - 0.01) > D_target:
                    rho = min(rho * sigma, 10.0)
            else:
                # Pretraining stage: basic VQ-VAE objective
                total_loss = recon_loss + vq_loss - 0.5 * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_recon += recon_loss.item()
            running_vq += vq_loss.item()
            running_pred += pred_loss.item()
            running_entropy += entropy_loss.item()
            batches += 1
        
        if epoch >= pretrain_epochs:
            scheduler.step()

        avg_recon = running_recon / batches
        avg_vq = running_vq / batches
        avg_pred = running_pred / batches
        avg_ent = running_entropy / batches

        # Log to console
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f} | Pred: {avg_pred:.4f} | Ent: {avg_ent:.4f}"
        )

        # Append to log file
        with open(output_file, "a") as f:
            f.write(
                f"Epoch {epoch + 1}: Recon={avg_recon:.6f}, VQ={avg_vq:.6f}, "
                f"Pred={avg_pred:.6f}, Ent={avg_ent:.6f}\n"
            )

    return model


############################################
# Extract and Reindex Encodings from Trained Model
############################################
def return_zq_list(
    X: torch.Tensor,
    model: EC_VQVAE,
    device: torch.device,
    time_steps: int,
    kernel_size: int,
    batch_size: int = 512,
) -> Tuple[List[int], float]:
    """
    Splits a long sequence X into overlapping chunks, runs through the model,
    and returns reindexed discrete codes along with average reconstruction loss.

    Args:
        X (torch.Tensor): Full sequence of shape (Total_T, C).
        model (EC_VQVAE): Trained model.
        device (torch.device): "cpu" or "cuda".
        time_steps (int): Number of time steps per block.
        kernel_size (int): Kernel size used in Encoder/Decoder padding.
        batch_size (int): Number of blocks per batch.

    Returns:
        reindexed_list (List[int]): Discrete codes reindexed to consecutive ints.
        avg_recon_loss (float): Average reconstruction loss over all blocks.
    """
    model.eval()
    Total_T, C = X.shape
    step = time_steps - (kernel_size - 1)
    blocks = []

    # Create overlapping windows
    start = 0
    while start + time_steps <= Total_T:
        end = start + time_steps
        blocks.append(X[start:end, :])
        start += step

    # Process in batches
    avg_recon_loss = 0.0
    all_codes = []
    num_blocks = len(blocks)
    with torch.no_grad():
        for i in range(0, num_blocks, batch_size):
            batch_blocks = blocks[i : i + batch_size]
            x_batch = torch.stack(batch_blocks, dim=0).to(device)  # (B, T, C)
            x_recon, _, _, _, encoding_indices = model(x_batch)
            recon_loss = F.mse_loss(x_recon, x_batch) * C
            avg_recon_loss += recon_loss.item() * (len(batch_blocks) / num_blocks)
            all_codes.extend(encoding_indices.cpu().view(-1).tolist())

    # Reindex codes to consecutive integers
    mapping = {}
    reindexed_list = []
    next_idx = 0
    for code in all_codes:
        if code not in mapping:
            mapping[code] = next_idx
            next_idx += 1
        reindexed_list.append(mapping[code])

    return reindexed_list, avg_recon_loss
