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
from torch.utils.data import DataLoader
from homework.models import MLPPlanner, TransformerPlanner, load_model, save_model
from homework.datasets.road_dataset import RoadDataset

def train(model_name, epochs=20, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Create model
    if model_name == "mlp_planner":
        model = MLPPlanner()
    elif model_name == "transformer_planner":
        model = TransformerPlanner()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Load dataset
    train_dataset = RoadDataset('drive_data/train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            waypoints = batch['waypoints'].to(device)
            
            # Forward pass
            pred = model(track_left, track_right)
            loss = loss_fn(pred, waypoints)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    # Save model
    save_model(model)
    print(f"Model {model_name} saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlp_planner", choices=["mlp_planner", "transformer_planner"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(args.model, args.epochs, args.batch_size, args.lr)