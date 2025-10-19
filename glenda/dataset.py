import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from .transforms import get_train_tfms, get_test_tfms

class GlendaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size=320, train=True, img_root=""):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.img_root = img_root or ""
        self.tfms = get_train_tfms(img_size) if train else get_test_tfms(img_size)

    def __len__(self):
        return len(self.df)

    def _read_img(self, p):
        p = os.path.join(self.img_root, p) if self.img_root else p
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {p}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._read_img(row["image_path"])
        aug = self.tfms(image=img)
        img = aug["image"]
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {"image": img, "label": label, "path": row["image_path"]}
