import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RoadSegmentationDataset
from model import UNet
from tqdm import tqdm

# Parâmetros
IMAGE_DIR = "images"
MASK_DIR = "masks"
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    dataset = RoadSegmentationDataset(IMAGE_DIR, MASK_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "unet_model.pth")

if __name__ == "__main__":
    train()
