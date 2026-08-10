"""
2_train.py

Trains an image classifier using transfer learning (a ResNet18 pretrained on ImageNet, fine-tuned on your dataset). Transfer learning is used deliberately here rather than training 
from scratch - it trains faster and performs better with a smaller dataset, which is realistic for a portfolio-scale project. 

Run with: pyhton scripts/2_train.py
"""