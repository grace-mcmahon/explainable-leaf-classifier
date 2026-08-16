# Explainable Leaf Disease Classifier

A plant disease classifier that predicts and shows its reasoning. 99% test accuracy, verified with Grad-CAM explainability to confirm the model is learning the right signal, not cheating on a lab-clean dataset.

## 1. What Problem Did I Solve?

Plant disease destroys a significant share of crops worldwide, with manual field inspection failing to scale. The challenge is to build an automated system that can reliably classify a leaf as healthy or diseased based on a single photo (and identify which disease), with a visual check as to *why* it made that decision — can I trust the reasoning, not just the accuracy number it reports?

A high accuracy score is poor evidence of a well-reasoned model on its own, since it could be exploiting some accidental shortcut in the data. This project is therefore inspectable end-to-end, not just performant.

## 2. Tools & Skills Used

- **Language:** Python
- **ML framework:** PyTorch, torchvision (transfer learning, ResNet18)
- **Explainability:** Grad-CAM (`pytorch-grad-cam`)
- **Evaluation:** scikit-learn (classification report, confusion matrix)
- **Data handling:** NumPy, PIL
- **Visualisation:** Matplotlib
- **Tooling:** Git, GitHub, virtual environments (venv)
