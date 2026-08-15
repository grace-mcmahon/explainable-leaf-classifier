"""
3_evaluate.py

Loads the best saved model and evaluates it properly on the held-out test set (data never seen during training or validation), producing a classification
report and confusion matrix. This is what gives you the real, honest, quotable  accuracy number for your README.

Run with: python scripts/3_evaluate.py
"""

import torch
import json
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path 

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("outputs/checkpoints")
OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main ():
    # load the class names saved durign training, so predictions map back to real labels 
    with open(CHECKPOINT_DIR / "classes.json") as f:
        classes = json.load(f)
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    #rebuild the same architecture, then load the traine weights into it
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_geatures, len(classes))
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval() # turns off dropout etc. - important for evaluation

    all_preds, all_labels = [], []
    with torch.no_grad(): # no need to track gradients, we're not training
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    print("\n=== Classification Report ===")
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(report)

    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # confusion matrix shows exactly which classes get mixed up with which 
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    print(f"\nConfusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"Classification report saved to {OUTPUT_DIR / 'classification_report.txt'}")

if __name__ == "__main__":
    main()


