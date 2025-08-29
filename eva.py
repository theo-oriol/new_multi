import os
from typing import List, Tuple, Optional
import pandas as pd
import json 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MultilabelF1Score
from torchvision.transforms.functional import pad


import timm
from timm.data import resolve_model_data_config, create_transform

NAME = "EVA"
MODEL_NAME   = "eva02_large_patch14_448.mim_in22k_ft_in22k"
NUM_CLASSES  = 15
BATCH_SIZE   = 2
EPOCHS       = 2
LR           = 1e-4
WEIGHT_DECAY = 0.05
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR     = "runs/runs_eva"  


CONFIG = {
    "name": NAME,
    "model_name": MODEL_NAME,
    "num_classes": NUM_CLASSES,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "device": DEVICE,
    "save_dir": SAVE_DIR,
    "criterion": "Focal_binary_cross_entropy(gamma=2)",
}

def startup_dir(name):
    i=0
    while os.path.isdir(os.path.join(SAVE_DIR,f"{name}_multi_{i}")) : i+=1
    destination_dir = os.path.join(SAVE_DIR,f"{name}_multi_{i}")
    os.makedirs(destination_dir)
    return destination_dir

def import_data(path):
    df = pd.read_csv(path)
    df["labels"] = df["labels"].apply(json.loads)
    X = df["name_of_img"].values        
    y = np.array(df["labels"].tolist())
    return X, y

class PathsAndLabels(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None, root: Optional[str]=None):
        assert len(paths) == len(labels), "paths and labels must match"
        self.paths = paths
        self.labels = labels

        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        p = self.paths[idx]
        if self.root and not os.path.isabs(p):
            p = os.path.join(self.root, p)
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[idx])  
        return img, label, rel_path


def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        return pad(img, get_padding(img), self.fill, self.padding_mode)
    
def make_transforms(model, is_train: bool):
    cfg = resolve_model_data_config(model)
    base = create_transform(**cfg, is_training=is_train,crop_pct=1.0)  
    if is_train:
        return T.Compose([NewPad(),T.RandomHorizontalFlip(p=0.5), base])
    return base



def build_model(num_classes: int):
    return timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)

class Focal_binary_cross_entropy(nn.Module):
    def __init__(self, gamma=2):
        super(Focal_binary_cross_entropy, self).__init__()
        self.gamma = gamma
    def forward(self, logits, targets):   
        targets = targets.to(dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce) 
        focal_weight = (1 - pt).pow(self.gamma)

        loss = focal_weight * bce
        loss = loss.mean()
        return loss


def train_simple(
    train_paths: List[str],
    train_labels: List[int],
    valid_paths: List[str],
    valid_labels: List[int],
    root: Optional[str] = None,
):
    model = build_model(NUM_CLASSES).to(DEVICE)

    tfm_train = make_transforms(model, is_train=True)
    tfm_val   = make_transforms(model, is_train=False)

    ds_train = PathsAndLabels(train_paths, train_labels, transform=tfm_train, root=root)
    ds_val   = PathsAndLabels(valid_paths, valid_labels, transform=tfm_val,   root=root)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = Focal_binary_cross_entropy()

    train_metric = MultilabelF1Score(num_labels=NUM_CLASSES, average="macro", threshold=0.5).to(DEVICE)
    val_metric   = MultilabelF1Score(num_labels=NUM_CLASSES, average="macro", threshold=0.5).to(DEVICE)

    
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    for epoch in range(1, EPOCHS + 1):
        tr_loss = 0.0
        va_loss = 0.0
        model.train()
        train_metric.reset()

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            train_metric.update(logits.sigmoid(), labels)
        scheduler.step()

        tr_loss = tr_loss / max(1, len(train_loader))
        tr_f1 = train_metric.compute()
        print(f"[Epoch {epoch:02d}] train loss {tr_loss:.4f} | F1 {tr_f1:.4f}", end="")

        model.eval()
        val_metric.reset()
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                loss = criterion(logits, labels)
                va_loss += loss.item()
                val_metric.update(logits.sigmoid(), labels)

        va_loss = va_loss/len(val_loader)
        va_f1 = val_metric.compute()
        print(f" || val loss {va_loss:.4f} | F1 {va_f1:.4f}")
        
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_f1"].append(tr_f1)
        hist["val_f1"].append(va_f1)

        model.eval()
        all_paths = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels, rel_paths in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                probs = logits.sigmoid().cpu().numpy()     # (B, C)
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())    # (B, C)
                all_paths.extend(rel_paths)                # list length += B

        all_probs  = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        prob_cols  = [f"prob_{i}"  for i in range(NUM_CLASSES)]
        label_cols = [f"label_{i}" for i in range(NUM_CLASSES)]

        df_out = pd.DataFrame({
            "image_path": all_paths
        })
        df_out[prob_cols]  = pd.DataFrame(all_probs,  index=df_out.index)
        df_out[label_cols] = pd.DataFrame(all_labels, index=df_out.index)

        csv_path = os.path.join(dir_path, "predictions_val.csv")
        df_out.to_csv(csv_path, index=False)

    return model, hist


if __name__ == "__main__":

    dir_path = startup_dir(NAME)

    with open(os.path.join(dir_path, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    list_of_train_image_path,list_of_train_image_labels = import_data("csv_split/habitatniveau1/train.csv")
    list_of_valid_image_path,list_of_valid_image_labels = import_data("csv_split/habitatniveau1/valid.csv")

    model, hist = train_simple(
        list_of_train_image_path, list_of_train_image_labels,
        list_of_valid_image_path, list_of_valid_image_labels,
        root="/home/oriol@newcefe.newage.fr/Datasets/whole_bird"  
    )

    torch.save(
        {"model": model.state_dict(), "num_classes": NUM_CLASSES, "model_name": MODEL_NAME},
        os.path.join(dir_path, "model.pth")
    )

    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(hist["epoch"], hist["train_f1"], label="train_F1(macro)")
    plt.plot(hist["epoch"], hist["val_f1"], label="val_F1(macro)")
    plt.xlabel("epoch"); plt.ylabel("F1 (macro)"); plt.title("Macro F1")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "f1_curve.png"), dpi=150)
    plt.close()


