#!/usr/bin/env python3
"""
Single-image inference. Feed any MRI image, get tumor class + confidence.

Usage:
    python predict.py --image path/to/mri.jpg
    python predict.py --image path/to/mri.jpg --weights models/best_model.pth
"""

import json
import argparse
import torch
from torchvision import transforms
from PIL import Image

from utils.model import TumorCNN
from utils.training_utils import get_device

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def predict(image_path, weights_path="models/best_model.pth", config_path="config.json"):
    with open(config_path) as f:
        config = json.load(f)

    device = get_device(config.get("device", "auto"), verbose=False)
    config["device"] = device

    # Load model
    model = TumorCNN(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load and preprocess image — accepts any size, any MRI format
    img = Image.open(image_path).convert("RGB")
    transform = get_transform(config["data"]["img_size"])
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 128, 128)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    print(f"\nImage : {image_path}")
    print(f"Result: {CLASSES[pred_idx].upper()}  ({probs[pred_idx]*100:.1f}% confidence)\n")
    print("All class probabilities:")
    for cls, prob in zip(CLASSES, probs):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12} {prob*100:5.1f}%  {bar}")
    print()

    return CLASSES[pred_idx], probs.cpu().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict tumor class from MRI image")
    parser.add_argument("--image",   required=True,              help="Path to MRI image")
    parser.add_argument("--weights", default="models/best_model.pth", help="Model weights")
    parser.add_argument("--config",  default="config.json",      help="Config file")
    args = parser.parse_args()

    predict(args.image, args.weights, args.config)
