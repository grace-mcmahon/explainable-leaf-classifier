"""
2_train.py

Trains an image classifier using transfer learning (a ResNet18 pretrained on ImageNet, fine-tuned on your dataset). Transfer learning is used deliberately here rather than training 
from scratch - it trains faster and performs better with a smaller dataset, which is realistic for a portfolio-scale project. 

Run with: pyhton scripts/2_train.py
"""

import torch #This is importing the core PyTorch library 
import torch.nn as nn #neural network building blocks (layers, loss functions)
from torch.utils.data import DataLoader  #handles batching and shuffling  data during training
from torchvision import datasets, transforms, models # Datasets/image tools, image processing, pretrained models
from pathlib import Path # For clean fiel path handling
from tqdm import tqdm # Provides progress bar in the terminal while training 
import json # For saving small pieces of data (like class names) to a file 

# We place the settings at the start here so we can easily find and adjust them 
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Let's create the folder if it already does not exist 

BATCH_SIZE = 32 # This is an assigned number for how mant images the model looks at per training step 
EPOCHS = 10 # How many full passes through the entire training set
LEARNING_RATE = 1e-4 # how big a step the model takes when learning from its mistakes

# If GPU not avaliable must fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders():
    """Prepares the training and validation data, ready to feed into the model. """
    #Transofrmations applied to TRAINING images: resize, then randomly flip/rotate
    # (this is "data augmentation" - it artificially creates variety,  which helps)
    # the model generalise instead of just memorising the exact training images
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet expects 224x224 images
        transforms.RandomHorizontalFlip(), # randomly flip left-right
        transforms.RandomRotation(15), # randomly rotate up to 15 degress
        transforms.ToTensor(), # convert image to a PyTorch tensor (numbers)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ^ These specific numbers are the average pizel values ResNet was
        # originially trained with on ImageNet - matching them helps the pretrianed weights work correctly on your images
    ])
    # validation images get resized and normalised the SAME way, but NO random flipping/rotating - you want to evaluate the image as it really is
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # ImageFolder auomatically reads your data.train and data/val folders, 
    # and treats each sub-folder name as a class label
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    # DataLoader wraps the dataset and hands out images in batches of BATCH_SIZE,
    # shuffling the training set each epoch (val doesnt need shuffling) 
    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

    return train_loader, val_loader, train_ds.classes  # classes = list of class names, e.g.["Tomato__healthy", ...]

def build_model(num_classes):
    """Loads a pretrained ResNet18 and adapts its final layer to our number of classes."""

    # Load ResNet18 with weights already trained on ImageNet (1000 general classes)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # ResNet's original final layer outputs 1000 numbers (one per ImageNet class).
    # We swap it for a new layer outputting exactly `num_classes` numbers instead —
    # this is the key trick in transfer learning: reuse everything the model
    # already learned about recognising shapes/edges/textures, just retrain
    # the final decision-making layer for OUR specific classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE) # move the model onto GPU or CPU, whichever DEVICE is

def run_epoch(model, loader, criterion, optimizer=None):
    """Runs one full pass through the data — either training (if optimizer given) or evaluating."""
    is_training = optimizer is not None 
    model.train() if is_training else model.eval()
    # .train() and .eval() switch the model's internal behaviour slightly
    # (some layers behave differently during raining vs evaluation)
    total_loss, correct, total = 0.0, 0, 0 
    
    # torch.enable_grad() / torch.no_grad() control whether PyTorch tracks
    # calculations for backpropogation - only needed during training, and 
    # skipping it during evaluation saves memory and runs faster 
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False): # tqdm just adds the progress bar
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_training:
                optimizer.zero_grad() # clear old gradients before computing new ones
            
            outputs = model(images) # run the images through the model - get predictions
            loss = criterion(outputs, labels) # copare predictions to true labels, get a 'how wrong' score

            if is_training:
                loss.backward() # compue how to adjust every weight to reduce the loss
                optimizer.step() # actually applythat adjustment
            
            # Track running totals so we can compute averahe loss/accuracy at the end
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1) # the class the model predicted (highest score)
            correct += (predicted == labels) .sum() .item()
            total += labels.size(0)
    return total_loss / total, correct / total # average loss, accuracy (0 to 1)

def main():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, classes = get_dataloaders()
    print(f"Classes ({len(classes)}): {classes}")
    # Save the class names to a file so other scripts (evaluate, gradcam) know
    # what the model's output nmber actually correspond to
    with open(OUTPUT_DIR / "classes.json", "w") as f:
        json.dump(classes, f)
    
    model = build_model(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Adam is a well-established, reliable choice of optimizer — it decides
    # HOW the model's weights get updated based on the loss
    best_val_acc = 0.0
    history = [] # We will record stats from every epoch here, for later reference

    for epoch in range(1, EPOCHS +1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        # notice: optimizer=None here means run_epoch treats this as EVALUATION, not training 
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss: .4f} val_acc={val_acc:.4f}")
        
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc" : train_acc,
                        "val_loss": val_loss, "val_acc": val_acc })
        # Only save the model when it's the best version so far - this means
        # you keep the version that generalises best, not just whatever
        # happened to finish last (which could slightly overfit)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            print(f" -> New best model saved (val_acc={val_acc: .4f})")
    # Save the full training history too, useful for writing up your README later
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Best val accuracy: {best_val_acc: .4f}")
    print(f"Best model saved to {OUTPUT_DIR / 'best_model.pth'}")

if __name__ == "__main__":
    main()






