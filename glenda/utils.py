import torch
import numpy as np
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def patient_level_split(df, val_frac=0.2, seed=42):
    pats = df["patient_id"].dropna().astype(str).unique()
    tr_pat, val_pat = train_test_split(pats, test_size=val_frac, random_state=seed, shuffle=True)
    tr = df[df["patient_id"].astype(str).isin(tr_pat)].reset_index(drop=True)
    va = df[df["patient_id"].astype(str).isin(val_pat)].reset_index(drop=True)
    return tr, va

def make_class_weights(df_train):
    counts = df_train["label"].value_counts().to_dict()
    w = torch.tensor([1.0/counts.get(0,1), 1.0/counts.get(1,1)], dtype=torch.float32)
    w = w / w.sum() * 2.0
    return w
