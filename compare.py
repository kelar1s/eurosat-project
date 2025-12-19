import os
import time
import torch
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms
from PIL import Image

from src.models import SimpleCNN, get_resnet18
from src.features import extract_hog_features
from src.dataset import get_eurosat_dataloaders

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

DEVICE = torch.device("cpu")


class ModelComparator:
    def __init__(self, data_dir="data", models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results = {}

        _, self.test_loader, _ = get_eurosat_dataloaders(data_dir)

    def evaluate_svm_hog(self):
        print("\n=== HOG + SVM ===")
        svm = joblib.load(os.path.join(self.models_dir, "svm_hog.pkl"))

        y_true, y_pred = [], []

        start = time.time()
        for imgs, labels in self.test_loader:
            for i in range(imgs.size(0)):
                feat = extract_hog_features(imgs[i])
                pred = svm.predict(feat)[0]
                y_pred.append(pred)
                y_true.append(labels[i].item())
        elapsed = time.time() - start

        acc = accuracy_score(y_true, y_pred)
        self.results["HOG + SVM"] = acc

        print(f"Accuracy: {acc:.4f}")
        print(f"Inference time: {elapsed:.2f}s")

    def evaluate_cnn(self):
        print("\n=== Simple CNN ===")
        model = SimpleCNN(10)
        model.load_state_dict(torch.load(os.path.join(self.models_dir, "simple_cnn.pth")))
        model.eval()

        y_true, y_pred = [], []

        start = time.time()
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                outputs = model(imgs)
                preds = outputs.argmax(1)
                y_pred.extend(preds.tolist())
                y_true.extend(labels.tolist())
        elapsed = time.time() - start

        acc = accuracy_score(y_true, y_pred)
        self.results["Simple CNN"] = acc

        print(f"Accuracy: {acc:.4f}")
        print(f"Inference time: {elapsed:.2f}s")

    def evaluate_resnet(self):
        print("\n=== ResNet18 ===")
        model = get_resnet18(10)
        model.load_state_dict(torch.load(os.path.join(self.models_dir, "resnet18.pth")))
        model.eval()

        y_true, y_pred = [], []

        start = time.time()
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                outputs = model(imgs)
                preds = outputs.argmax(1)
                y_pred.extend(preds.tolist())
                y_true.extend(labels.tolist())
        elapsed = time.time() - start

        acc = accuracy_score(y_true, y_pred)
        self.results["ResNet18"] = acc

        print(f"Accuracy: {acc:.4f}")
        print(f"Inference time: {elapsed:.2f}s")

        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=CLASSES))

    def summary(self):
        print("\n" + "=" * 60)
        print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 60)

        for name, acc in sorted(self.results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {acc:.4f}")

        best = max(self.results, key=self.results.get)
        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best}")

    def run(self):
        self.evaluate_svm_hog()
        self.evaluate_cnn()
        self.evaluate_resnet()
        self.summary()


if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.run()
