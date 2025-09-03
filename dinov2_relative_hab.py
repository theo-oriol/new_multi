import os
from typing import List, Optional
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import json 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MultilabelF1Score
from torchvision.transforms.functional import pad


NAME        = "DINOv2_relatif"
ARCH        = "dinov2_vits14"   # dinov2_vits14 / vitb14 / vitl14 / vitg14
NUM_CLASSES = 9
BATCH_SIZE  = 40
EPOCHS      = 2
LR          = 1e-4
WEIGHT_DECAY= 0.05
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "runs/runs_dino_relatif"  

CONFIG = {
    "name": NAME,
    "model_name": ARCH,
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
    i = 0
    while os.path.isdir(os.path.join(SAVE_DIR, f"{name}_multi_{i}")):
        i += 1
    destination_dir = os.path.join(SAVE_DIR, f"{name}_multi_{i}")
    os.makedirs(destination_dir)
    return destination_dir

def import_data(path):
    df = pd.read_csv(path)
    df["labels"] = df["labels"].apply(json.loads)
    X = df["name_of_img"].values
    y = np.array(df["labels"].tolist(), dtype=np.float32)  
    return X, y


class PathsAndLabels(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, transform=None, root: Optional[str] = None):
        assert len(paths) == len(labels), "paths and labels must match"
        self.paths = list(paths)
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


class DinoV2Classifier(nn.Module):
    def __init__(self, arch: str = ARCH, num_classes: int = NUM_CLASSES, train_backbone: bool = True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", arch)
        if "vits14" in arch:
            feat_dim = 384
        elif "vitb14" in arch:
            feat_dim = 768
        elif "vitl14" in arch:
            feat_dim = 1024
        elif "vitg14" in arch:
            feat_dim = 1536
        else:
            raise ValueError(f"Unknown DINOv2 arch: {arch}")

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)    
        logits = self.head(feats)         
        return logits

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
    

def make_transforms(is_train: bool):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    if is_train:
        return T.Compose([
            NewPad(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            NewPad(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

class BrierScore(torch.nn.Module):
    def __init__(self, num_classes, per_class=False ):
        super().__init__()
        self.num_classes = num_classes
        self.per_class = per_class
        self.add_state("sum", torch.zeros(num_classes if per_class else 1))
        self.add_state("total", torch.tensor(0.0))

    def add_state(self, name, default):
        setattr(self, name, default)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        device = self.sum.device
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.softmax(y_pred, dim=-1)

        loss = (y_pred - y_true) ** 2
        if self.per_class:
            self.sum += loss.sum(dim=0)
        else:
            self.sum += loss.sum()
        self.total += y_true.size(0)

    def compute(self):

        return self.sum / self.total
    
    def reset(self):
        if self.per_class:
            self.sum = torch.zeros(self.num_classes, device=self.sum.device)
        else:
            self.sum = torch.zeros(1, device=self.sum.device)
        self.total = torch.tensor(0.0, device=self.sum.device)
    

def train_simple(
    train_paths: List[str],
    train_labels: np.ndarray,
    valid_paths: List[str],
    valid_labels: np.ndarray,
    dir_path: str,
    root: Optional[str] = None,
):
    model = DinoV2Classifier().to(DEVICE)

    tfm_train = make_transforms(is_train=True)
    tfm_val   = make_transforms(is_train=False)

    ds_train = PathsAndLabels(train_paths, train_labels, transform=tfm_train, root=root)
    ds_val   = PathsAndLabels(valid_paths, valid_labels, transform=tfm_val,   root=root)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.KLDivLoss(reduction="batchmean")

    train_metric = BrierScore(num_classes=NUM_CLASSES).to(DEVICE)
    val_metric   = BrierScore(num_classes=NUM_CLASSES).to(DEVICE)

    tr_loss_accum = 0.0
    va_loss_accum = 0.0
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_brier": [], "val_brier": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_metric.reset()
        tr_loss_accum = 0.0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)  

            logits = model(imgs)
            loss = criterion(logits.sigmoid(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metric.update(logits.sigmoid(), labels) 
            
            tr_loss_accum += loss.item()
        scheduler.step()
        tr_loss = tr_loss_accum / max(1, len(train_loader))
        tr_brier = train_metric.compute().item()
        print(f"[Epoch {epoch:02d}] train loss {tr_loss:.4f} | Brier {tr_brier:.4f}", end="")

        model.eval()
        val_metric.reset()
        va_loss_accum = 0.0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                loss = criterion(logits.sigmoid(), labels)
                val_metric.update(logits.sigmoid(), labels)
                va_loss_accum += loss.item()

        va_loss = va_loss_accum / max(1, len(val_loader))
        va_brier = val_metric.compute().item()
        print(f" || val loss {va_loss:.4f} | val Brier {va_brier:.4f}")

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_brier"].append(tr_brier)
        hist["val_brier"].append(va_brier)


        model.eval()
        all_paths = []
        all_probs = []
        all_labels = []

    test_metric   = BrierScore(num_classes=NUM_CLASSES, per_class=True).to(DEVICE)
    test_metric.reset()
    with torch.no_grad():
        for imgs, labels, rel_paths in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            probs = logits.sigmoid().cpu().numpy()   
            test_metric.update(logits.sigmoid(), labels)  
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())    
            all_paths.extend(rel_paths)  
    test_brier = test_metric.compute()            

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


        


    return model, hist, test_brier


if __name__ == "__main__":
    dir_path = startup_dir(NAME)

    with open(os.path.join(dir_path, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    train_paths, train_labels = import_data("habitatniveau1_relatif/train.csv")
    valid_paths, valid_labels = import_data("habitatniveau1_relatif/valid.csv")

    model, hist, test_brier = train_simple(
        train_paths, train_labels,
        valid_paths, valid_labels,
        dir_path,
        root="/home/oriol@newcefe.newage.fr/Datasets/whole_bird",
    )

    torch.save({"model": model.state_dict(), "arch": ARCH, "num_classes": NUM_CLASSES},
               os.path.join(dir_path, "model.pth"))

    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "loss_curve.png"), dpi=150)
    plt.close()

   
    plt.figure()
    plt.plot(hist["epoch"], hist["train_brier"], label="train_BRIER(macro)")
    plt.plot(hist["epoch"], hist["val_brier"], label="val_BRIER(macro)")
    plt.xlabel("epoch"); plt.ylabel("BRIER (macro)"); plt.title("Macro BRIER")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "brier_curve.png"), dpi=150)
    plt.close()

    
    num_classes = test_brier.numel()
    values = test_brier.detach().cpu().numpy()


    with open("labels_habitat1_relatif.json", "r") as f:
        class_names = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.bar(range(num_classes), values, tick_label=class_names)
    plt.ylabel("Brier Score")
    plt.title("Per-class Brier Score")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(os.path.join(dir_path, "brier_multi_curve.png"), dpi=150)
    plt.close()


    