"""
4_gradcam.py

Generates Grad_CAM heatmaps showing which parts of each image the model
focused on when making its prediction. This is the explainability piece
that differentiates this project from a standard "trained a classifier"
portfolio piece - its lets you show, visually, WHY the model decided
what it decided, including on images it got wrong.

Run with: python scripts/4_gradcam.py
"""

import torch 
import json
import numpy as np 
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utiles.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import random

DATA_DIR = Path("data/test")
CHECKPOINT_DIR = Path("outputs/checkpoints")
OUTPUT_DIR = Path("outputs/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CORRECT_EXAMPLES = 3
N_INCORRECT_EXAMPLES = 3

def load_model(classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

def predict_and_visualise(model, img_path, true_class_idx, classes, cam, transform, tag):
    image = Image,open(img_path).convert("RGB")
    input_tensor = transform(image).unqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()

    tragets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(image.resize((224, 224))) / 255.0
    visualisation = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"True: {classes[true_class_idx]}")
    axes[0].axis("off")
    axes[1].imshow(visualisation)
    axes[1].set_title(f"Pred: {classes[pred_idx]}")
    axes[1].axis("off")
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{tag}_{img_path.stem}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return pred_idx == true_class_idx, out_path

def main():
    with open(CHECKPOINT_DIR / "classes.json") as f:
        classes = json.load(f)
    
    model = load_model(classes)
    target_layer = [model.layer4[-1]] # last conv block of ResNet18 - standard choice for Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    correct_found, incorrect_found = 0, 0
    class_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    all_pairs = []
    for class_idx, class_dir in enumerate(class_dirs):
        for img_path in class_dir.glob("*.*"):
            all_pairs.append((img_path, class_idx))
    random.shuffle(all_pairs) # random order so examples aren't all from one class

    for img_path, true_idx in all_pairs:
        if correct_found >= N_CORRECT_EXAMPLES and incorrect_found >= N_INCORRECT_EXAMPLES:
            break

        # Save with a temporary tag; rename once we know if it was actually correct.
        was_correct, out_path = predict_and_visualise(
            model, img_path, true_idx, classes, cam, transform, tag="tmp"
        )

        if was_correct and correct_found < N_CORRECT_EXAMPLES:
            correct_found += 1
            final_path = out_path.with_name(out_path.name.replace("tmp_", "correct_"))
            out_path.rename(final_path)
            print(f"Saved correct example: {final_path}")
        else:
            out_path.unlink() # already have enough of this category, discard
    
    print(f"\nDone. {correct_found} correct + {incorrect_found} incorrect examples saved to {OUTPUT_DIR}")
    print("Use these images in your README's Key Findings section.")

if __name__ == "__main__":
    main()
