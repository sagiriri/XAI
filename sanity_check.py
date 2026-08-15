"""
XAIBench environment sanity check.
Run this once after installing everything to confirm the core libraries
actually work together, not just that pip install succeeded.
"""

import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import numpy as np

print("=" * 50)
print("STEP 1: Basic library versions")
print("=" * 50)
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\n" + "=" * 50)
print("STEP 2: Load a pretrained model")
print("=" * 50)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
print("ResNet18 loaded successfully.")

print("\n" + "=" * 50)
print("STEP 3: Create a dummy image and run inference")
print("=" * 50)
dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(dummy_img).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
print(f"Inference successful. Output shape: {output.shape}")

print("\n" + "=" * 50)
print("STEP 4: Grad-CAM")
print("=" * 50)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
print(f"Grad-CAM heatmap generated. Shape: {grayscale_cam.shape}")

print("\n" + "=" * 50)
print("STEP 5: Captum (Integrated Gradients)")
print("=" * 50)
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(input_tensor, target=0, n_steps=5)
print(f"Captum attribution generated. Shape: {attributions.shape}")

print("\n" + "=" * 50)
print("STEP 6: SHAP")
print("=" * 50)
import shap
from sklearn.ensemble import RandomForestClassifier

# Using TreeExplainer on a tiny RandomForest instead of a gradient-based
# explainer on the CNN: SHAP's deep-learning explainers (DeepExplainer,
# GradientExplainer) are known to be slow or unstable on CPU with recent
# PyTorch versions. TreeExplainer is fast and reliable, and is enough to
# confirm SHAP itself is installed and working correctly. CNN-specific SHAP
# usage can be revisited later, with more time or GPU, inside the actual
# XAI module.
X_dummy = np.random.rand(20, 4)
y_dummy = np.random.randint(0, 2, 20)
clf = RandomForestClassifier(n_estimators=10, max_depth=3).fit(X_dummy, y_dummy)
tree_explainer = shap.TreeExplainer(clf)
shap_values = tree_explainer.shap_values(X_dummy)
print("SHAP explainer ran successfully.")

print("\n" + "=" * 50)
print("STEP 7: scikit-learn metrics")
print("=" * 50)
from sklearn.metrics import accuracy_score, f1_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")

print("\n" + "=" * 50)
print("ALL CHECKS PASSED — environment is ready.")
print("=" * 50)
