"""I have used Gitlab Copilot for assistance"""
"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import PlannerMetric

def train(model_name, epochs=20, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Create model
    if model_name == "mlp_planner":
        model = MLPPlanner()
    elif model_name == "transformer_planner":
        model = TransformerPlanner()
    elif model_name == "cnn_planner":
        model = CNNPlanner()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    loss_fn = nn.MSELoss()
    
    # Load dataset - all episodes
    import os
    from torch.utils.data import ConcatDataset
    
    data_dir = 'drive_data/train'
    
    # Get all episode folders
    episodes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(episodes)} episodes")
    
    # Load each episode
    datasets = []
    # Use "default" transform for CNN (needs images), "default" for MLP/Transformer
    train_transform = "default"
    for episode in sorted(episodes):
        episode_path = os.path.join(data_dir, episode)
        try:
            dataset = RoadDataset(episode_path, transform_pipeline=train_transform)
            datasets.append(dataset)
        except Exception as e:
            print(f"Warning: Failed to load {episode}: {e}")
    
    # Combine all episodes
    train_dataset = ConcatDataset(datasets)
    print(f"Total samples: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Load validation dataset
    val_dir = 'drive_data/val'
    val_episodes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    val_datasets = []
    # Use "default" transform for CNN (needs images), "state_only" for MLP/Transformer
    val_transform = "default" if model_name == "cnn_planner" else "state_only"
    for episode in sorted(val_episodes):
        episode_path = os.path.join(val_dir, episode)
        try:
            dataset = RoadDataset(episode_path, transform_pipeline=val_transform)
            val_datasets.append(dataset)
        except Exception as e:
            print(f"Warning: Failed to load val {episode}: {e}")
    
    val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training loop
    best_lat_error = float('inf')
    patience = 30  # Increased from 15 to allow more training
    no_improve_count = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch in train_loader:
            waypoints = batch['waypoints'].to(device)
            
            # Forward pass (different inputs for CNN vs MLP/Transformer)
            if model_name == "cnn_planner":
                image = batch['image'].to(device)
                pred = model(image)
            else:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                pred = model(track_left, track_right)
            # Calculate loss (use model's custom loss if available)
            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(pred, waypoints)
            else:
                loss = loss_fn(pred, waypoints)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        metrics = PlannerMetric()
        with torch.no_grad():
            for batch in val_loader:
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                
                # Forward pass (different inputs for CNN vs MLP/Transformer)
                if model_name == "cnn_planner":
                    image = batch['image'].to(device)
                    pred = model(image)
                else:
                    track_left = batch['track_left'].to(device)
                    track_right = batch['track_right'].to(device)
                    pred = model(track_left, track_right)
                metrics.add(pred, waypoints, waypoints_mask)
        
        results = metrics.compute()
        lat_err = results["lateral_error"]
        
        print(f'Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | '
              f'Long.Err: {results["longitudinal_error"]:.4f} | '
              f'Lat.Err: {lat_err:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping
        if lat_err < best_lat_error:
            best_lat_error = lat_err
            no_improve_count = 0
            save_model(model)
            print(f"  ✓ Best model saved! Lat.Err: {lat_err:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlp_planner", choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(args.model, args.epochs, args.batch_size, args.lr)