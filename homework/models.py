from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_size = n_track * 2 * 2
        output_size = n_waypoints * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        # Separate heads for longitudinal and lateral predictions
        self.lon_head = nn.Linear(32, n_waypoints)
        self.lat_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_waypoints),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        
        x = torch.cat([track_left, track_right], dim=1)

        b = x.shape[0]
        x = x.reshape(b, -1)
        # Pass through shared MLP
        x = self.mlp(x)
        
        # Separate predictions for lon and lat
        lon_pred = self.lon_head(x)  # (b, n_waypoints)
        lat_pred = self.lat_head(x)  # (b, n_waypoints)
        
        # Combine [lon, lat] into waypoints
        x = torch.stack([lon_pred, lat_pred], dim=2)  # (b, n_waypoints, 2)

        return x


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 256,  # Increased capacity
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.track_encoder = nn.Linear(2, d_model)
        
        # Add positional encoding for track positions
        self.pos_encoder = nn.Embedding(2 * n_track, d_model)
        
        # Add normalization
        self.track_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.3,  # Increased to allow better regularization
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2,  # More layers for deeper processing
        )

        # Separate heads for better lon/lat learning
        self.lon_head = nn.Linear(d_model, 1)
        self.lat_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.shape[0]
        
        # Concatenate left and right tracks
        x = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        
        # Encode track points
        encoded_tracks = self.track_encoder(x)  # (b, 2*n_track, d_model)
        
        # Add positional encoding
        positions = torch.arange(2*self.n_track, device=x.device).unsqueeze(0).expand(b, -1)
        pos_embeds = self.pos_encoder(positions)
        encoded_tracks = encoded_tracks + pos_embeds
        encoded_tracks = self.track_norm(encoded_tracks)  # Normalize
        
        # Create query embeddings for waypoints
        waypoint_queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        waypoint_queries = self.query_norm(waypoint_queries)  # Normalize
        
        # Cross-attention: waypoints attend to track boundaries
        output = self.transformer_decoder(
            tgt=waypoint_queries,
            memory=encoded_tracks,
        )
        
        # Project to 2D waypoints with separate heads
        lon_pred = self.lon_head(output)  # (b, n_waypoints, 1)
        lat_pred = self.lat_head(output)  # (b, n_waypoints, 1)
        
        # Combine into waypoints
        waypoints = torch.cat([lon_pred, lat_pred], dim=2)  # (b, n_waypoints, 2)
        
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # Loss weights: prioritize longitudinal error [1.5, 1.0]
        self.register_buffer("loss_weights", torch.tensor([1.5, 1.0]), persistent=False)

        # CNN backbone: input (B, 3, 96, 128) -> features
        self.backbone = nn.Sequential(
            # Block 1: 3 -> 64 channels (increased capacity)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 48, 64)
            
            # Block 2: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 128, 24, 32)
            
            # Block 3: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 256, 12, 16)
            
            # Block 4: 256 -> 256 channels
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 256, 6, 8)
            
            # Adaptive pooling to flatten to fixed size
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 256, 1, 1)
        )
        
        # Dense layers for feature processing (increased capacity)
        self.dense = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        
        # Separate heads for longitudinal and lateral predictions
        self.lon_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_waypoints),
        )
        self.lat_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_waypoints),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Normalize input
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # CNN backbone
        x = self.backbone(x)  # (B, 256, 1, 1)
        
        # Flatten
        x = x.view(x.shape[0], -1)  # (B, 256)
        
        # Dense layers
        x = self.dense(x)  # (B, 128)
        
        # Separate heads for lon and lat
        lon_pred = self.lon_head(x)  # (B, n_waypoints)
        lat_pred = self.lat_head(x)  # (B, n_waypoints)
        
        # Combine [lon, lat] into waypoints
        waypoints = torch.stack([lon_pred, lat_pred], dim=2)  # (B, n_waypoints, 2)
        
        return waypoints
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss, emphasizing longitudinal error.
        
        Args:
            pred (torch.Tensor): predicted waypoints (B, n_waypoints, 2)
            target (torch.Tensor): target waypoints (B, n_waypoints, 2)
        
        Returns:
            torch.Tensor: weighted MSE loss
        """
        mse = (pred - target) ** 2  # (B, n_waypoints, 2)
        weighted_mse = mse * self.loss_weights.unsqueeze(0).unsqueeze(0)
        return weighted_mse.mean()


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
