import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ── 설정 ──────────────────────────────────────────
DATA_DIR   = "../data/animal_data/cropped" # You need unzip
SAVE_PATH  = "./animal_predict_model.pth"
BATCH_SIZE = 8
NUM_EPOCHS = 10
LR         = 1e-4
VAL_RATIO  = 0.2
# ──────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def main():
    # ImageFolder로 자동 라벨링 (폴더명 = 클래스명)
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    classes = full_dataset.classes
    print(f"클래스: {classes}")
    print(f"전체 데이터: {len(full_dataset)}장")

    # train / val 분리
    val_size   = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"train: {train_size}장 / val: {val_size}장")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 모델
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 학습
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        # train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | loss: {total_loss/len(train_loader):.4f} | train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

        # best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → best 모델 저장 (val acc: {val_acc:.4f})")

    print(f"\n학습 완료! best val acc: {best_val_acc:.4f}")
    print(f"모델 저장: {SAVE_PATH}")


if __name__ == "__main__":
    main()
