import os, argparse, yaml
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from glenda.dataset import GlendaDataset
from glenda.models import make_model, get_default_target_layer

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num", type=int, default=10, help="number of positive predictions to visualise")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["gradcam"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Load validation split for quick visualisation (re-using the same split logic)
    df = pd.read_csv(cfg["data"]["metadata_csv"])    
    # Simple random sample for demo (no masks required)
    df = df.sample(frac=1.0, random_state=cfg["training"].get("seed",42)).reset_index(drop=True)

    ds = GlendaDataset(df, img_size=cfg["training"]["img_size"], train=False, img_root=cfg["data"].get("img_root",""))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = make_model(cfg["model"]["name"], num_classes=2, pretrained=False).to(device)
    ck = f"checkpoints/{cfg['model']['name']}_best.pt"
    if not os.path.exists(ck):
        raise FileNotFoundError("Checkpoint not found. Please run train.py first.")
    model.load_state_dict(torch.load(ck, map_location=device))
    model.eval()

    target_layer = get_default_target_layer(model, cfg["model"]["name"])
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device=="cuda"))

    saved = 0
    for i, batch in enumerate(tqdm(dl, desc="Grad-CAM")):
        img = batch["image"].to(device)
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0,1].item()
        pred = int(probs >= 0.5)

        if pred == 1:  # focus on predicted positives
            grayscale_cam = cam(input_tensor=img)[0]  # HxW in [0,1]
            # unnormalize: from [-1,1] back to [0,255]
            rgb = (img[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5) * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            overlay = show_cam_on_image(rgb.astype(np.float32)/255.0, grayscale_cam, use_rgb=True)
            out_path = os.path.join(out_dir, f"cam_{i:04d}_p{probs:.2f}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= args.num:
                break

    print(f"Saved {saved} Grad-CAM overlays to {out_dir}")

if __name__ == "__main__":
    main()
