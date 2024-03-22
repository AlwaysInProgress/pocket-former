from turn_classifier import TurnClassifier
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from mg import MGDataset, DataPipeline
from torch.utils.data import DataLoader
import datetime
from utils import *
import os
import cv2
import numpy as np
from torch.utils.data import RandomSampler

def train_epoch(model, optimizer, args, train_loader, device):
    model.train()
    loss_sum = 0

    for batch, labels in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.to(device)
        output = model(batch)
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
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--frames_per_item", type=int, default=2)
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TurnClassifier(hidden_dim=1024, num_classes=2, enc_name="vit-base", num_frames=args.frames_per_item)
    model.to(device)
    print(model)

    if args.train:
        pipeline = DataPipeline(frames_per_item=args.frames_per_item)
        train_dataset = pipeline.get_dataset('train')
        val_dataset = pipeline.get_dataset('test')
        print("train dataset length: ", len(train_dataset))
        print("val dataset length: ", len(val_dataset))
        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset) * 0.25))
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)
        train(model, args, train_loader, val_loader, device)
    elif args.live:
        model.eval()
        # live inference
        with torch.no_grad():
            cap = cv2.VideoCapture(0)
            while True:
                images = []
                for i in range(2):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # preserving aspect ratio
                    frame = cv2.resize(frame, (frame.shape[1] * 224 // frame.shape[0], 224))
                    frame = center_crop(frame, (224, 224))
                    images.append(frame)
                images = np.array(images)
                images = images / 255.0
                images = torch.tensor(images).permute(0, 3, 1, 2).float().unsqueeze(0)
                images = images.to(device)
                output = model(images)
                # print("output: ", output)
                viz_mg_data((images[0], None), model, live=True)
    else:
        pipeline = DataPipeline(frames_per_item=args.frames_per_item)
        val_dataset = pipeline.get_dataset('test')
        viz_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()

        # visualizing image and output
        with torch.no_grad():
            for batch, labels in viz_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                output = model(batch)

                # visualizing the 0th image
                viz_mg_data((batch[0], labels[0]), model)