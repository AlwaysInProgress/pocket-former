from turn_classifier import TurnClassifier
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
from mg import MGDataset
from torch.utils.data import DataLoader
import datetime
import cv2
from utils import *

def train_epoch(model, optimizer, args, train_loader, device):
    model.train()
    loss_sum = 0

    for batch, labels in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.to(device)
        output = model(batch)
        print("batch shape: ", batch.shape)
        print("output shape: ", output.shape)
        print("labels shape: ", labels.shape)
        print("labels: ", labels)
        loss = nn.functional.cross_entropy(output, labels)
        print("loss: ", loss.item())
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    loss_avg = loss_sum / len(train_loader)

    return loss_avg

def eval_epoch(model, args, val_loader, device):
    model.train()
    loss_sum = 0

    with torch.no_grad():
        for batch, labels in tqdm(val_loader):
            batch = batch.to(device)
            labels = labels.to(device)
            output = model(batch)
            loss = nn.functional.cross_entropy(output, labels)
            loss_sum += loss.item()

    loss_avg = loss_sum / len(val_loader)

    return loss_avg

def train(model, args, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, args, train_loader, device)
        val_loss = eval_epoch(model, args, val_loader, device)
        print(f"Epoch {epoch} train loss: {train_loss}, val loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(model.state_dict(), f"checkpoints/best_turn_classifier_{datetime_str}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    train_loader = DataLoader(MGDataset(frames_per_item=2, split='train'), batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(MGDataset(frames_per_item=2, split='test'), batch_size=args.bs, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurnClassifier(hidden_dim=1024, num_classes=2, enc_name="vit-base", num_frames=2)
    model.to(device)
    print(model)

    if args.train:
        train(model, args, train_loader, val_loader, device)
    else:
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()

        # visualizing image and output
        with torch.no_grad():
            for batch, labels in val_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                output = model(batch)

                # visualizing the 0th image
                viz_mg_data((batch[0], labels[0]), model)