from turn_classifier import TurnClassifier
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse

def train_epoch(model, optimizer, args, train_loader, device):
    model.train()
    loss_sum = 0

    for batch, labels in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.view(-1).to(device)
        output = model(batch).view((-1, model.vocab_size))
        probs = torch.softmax(output, dim=-1)
        loss = nn.functional.cross_entropy(output, labels)
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
            labels = labels.view(-1).to(device)
            output = model(batch).view((-1, model.vocab_size))
            loss = nn.functional.cross_entropy(output, labels)
            loss_sum += loss.item()

    loss_avg = loss_sum / len(val_loader)

    return loss_avg

def train(model, args, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, args, train_loader, device)
        val_loss = eval_epoch(model, args, val_loader, device)
        print(f"Epoch {epoch} train loss: {train_loss}, val loss: {val_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_loader = None # TODO: make this
    val_loader = None # TODO: make this

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurnClassifier(hidden_dim=1024, num_classes=2, enc_name="vit-base", num_frames=2).to(device)
    print(model)
    x = np.random.randint(0, 256, (2, 2, 224, 224, 3))
    train(model, args, train_loader, val_loader, device)
    res = model(x)
    print(res)