**Explainable Leaf Disease Classifier**
A plant disease classifier that predicts and shows its reaosning. 99% test accuracy, verified with Grad-CAM explainability to confirm the model is learning the right signal, not cheating on a lab-clean dataset. 

*1. What Problem Did I Solve?*
Plant disease destroys a significant share fo crops worldwide, with manual field inspection failing to scale. The challenge is to build an automated system that can reliably classify a leaf as healthy or diseased based on a single photo (and identify which disease), and a visual check as to why it made that decision (ie., can I trust *why* it's making that call, not just the accuracy number it reports?)

A high accuracy score is solely poor evidence of a well reasned model, as it could be exploiting some accidental shortcut in the data. This project is therefore inspectable end-to-end, not just performant.

*2. Tools & Skills Used*
**Language:** Python
**ML framework:** PyTorch, torchvision (transfer learning, ResNet18)
**Explainability:** GradCAM (pytorch-grad-cam)
**Evaluation:** scikit-learn (classification report, confusion matrix)
**Data handling:** NumPy, PIL
**Visualisation:** Matplotlib
**Tooling:** Git, GitHub, virtual environment (venv)

*3. How I Approached It*

*4. Where Is the Code*

*Run It Yourself*

*5. What Was the Outcome*

*Data Quality Notes*

*What I'd Do Next*
