import timm

def make_model(name="edgenext_small", num_classes=2, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

def get_default_target_layer(model, model_name="edgenext_small"):
    # For EdgeNeXt, using the last stage works reasonably for CAM
    return getattr(model, "stages")[-1] if hasattr(model, "stages") else None
