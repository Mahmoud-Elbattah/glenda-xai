# Explainable Deep Learning for Endometriosis Classification in Laparoscopic Images

This is a **minimal** PyTorch + timm pipeline that trains a **single best model (EdgeNeXt_Small)**
for binary classification on GLENDA and provides a **Grad-CAM** visualisation script.

Zhu, Y., & Elbattah, M. (2025). Explainable Deep Learning for Endometriosis Classification in Laparoscopic Images. BioMedInformatics, 5(4), 63.
https://doi.org/10.3390/biomedinformatics5040063

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --config config.yaml
python gradcam_demo.py --config config.yaml --num 10  # save 10 Grad-CAM overlays
```

### Expected `metadata.csv` columns
- `image_path` : path to RGB image (relative or absolute)
- `label`      : 0 (normal), 1 (pathological)
- `patient_id` : string/int used to build patient-level split

*(No `mask_path` is needed for this lightweight demo.)*

## License
Code released under **CC BY 4.0** — see `LICENSE`.
