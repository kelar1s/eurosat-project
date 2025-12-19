import torch
import torch.nn as nn
from sklearn.svm import SVC
import joblib
from tqdm import tqdm
from .features import extract_hog_features

def train_svm_hog(train_db, save_path="models/svm_hog.pkl"):
    """
    Обучает SVM по HOG-признакам.
    """
    X, y = [], []
    for i in range(min(len(train_db), 1500)):
        img, label = train_db[i]
        feat = extract_hog_features(img)
        X.append(feat.flatten())
        y.append(label)

    clf = SVC(probability=True, kernel='rbf')
    clf.fit(X, y)
    joblib.dump(clf, save_path)
    print(f"SVM+HOG модель сохранена в: {save_path}")


def train_deep(model, train_loader, save_path, device, lr=1e-3, epochs=5):
    """
    Обучение нейросетевой модели.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена: {save_path}")
