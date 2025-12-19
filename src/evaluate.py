import torch
from sklearn.metrics import accuracy_score

def evaluate_dl_model(model, dataloader, device):
    model.to(device).eval()
    preds, truths = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds.extend(out.argmax(1).cpu().numpy())
            truths.extend(labels.numpy())
    return accuracy_score(truths, preds)
