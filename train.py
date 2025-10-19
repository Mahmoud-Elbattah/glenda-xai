import os, argparse, yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from glenda.dataset import GlendaDataset
from glenda.models import make_model
from glenda.utils import set_seed, patient_level_split, make_class_weights

def build_loss(cfg, class_weights=None):
    if cfg["training"]["loss"] == "weighted_ce" and class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()

def evaluate(loader, model, device):
    model.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:,1]
            probs_all.append(probs.cpu())
            y_all.append(y.cpu())
    probs_all = torch.cat(probs_all).numpy()
    y_all = torch.cat(y_all).numpy()
    preds = (probs_all >= 0.5).astype(int)
    acc = (preds == y_all).mean()
    return float(acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(cfg["data"]["metadata_csv"])    
    tr_df, va_df = patient_level_split(df, val_frac=cfg["training"]["val_split"], seed=cfg["training"].get("seed",42))

    bs = cfg["training"]["batch_size"]
    img_size = cfg["training"]["img_size"]
    ds_tr = GlendaDataset(tr_df, img_size=img_size, train=True,  img_root=cfg["data"].get("img_root",""))
    ds_va = GlendaDataset(va_df, img_size=img_size, train=False, img_root=cfg["data"].get("img_root",""))

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=cfg["data"]["num_workers"])    
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = make_model(cfg["model"]["name"], num_classes=2, pretrained=True).to(device)

    cw = make_class_weights(tr_df).to(device) if cfg["training"]["loss"] == "weighted_ce" else None
    criterion = build_loss(cfg, class_weights=cw)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["max_epochs"])

    best_va, best_state, patience = 0.0, None, cfg["training"]["patience"]
    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        for batch in tqdm(dl_tr, desc=f"Epoch {epoch+1}"):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        va_acc = evaluate(dl_va, model, device)
        if va_acc > best_va + 1e-5:
            best_va = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = cfg["training"]["patience"]
        else:
            patience -= 1
            if patience == 0: break

    if best_state is not None:
        model.load_state_dict({k: v for k, v in best_state.items()})
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{cfg['model']['name']}_best.pt")
    print(f"Done. Best val acc={best_va:.3f}" )

if __name__ == "__main__":
    main()
