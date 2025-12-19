import os
import torch
from torchvision import datasets, transforms

def get_eurosat_dataloaders(data_dir: str, batch_size: int = 32, img_size=(64, 64)):
    """
    Загружает датасет EuroSAT RGB с помощью ImageFolder.
    Возвращает train и test DataLoader.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    class EuroSatRGB(datasets.ImageFolder):
        def find_classes(self, directory):
            classes, class_to_idx = super().find_classes(directory)
            if 'allBands' in classes:
                classes.remove('allBands')
                del class_to_idx['allBands']
            return classes, class_to_idx

    full_dataset = EuroSatRGB(root=data_dir, transform=transform)
    total = len(full_dataset)
    train_size = int(0.8 * total)
    test_size = total - train_size

    train_db, test_db = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_db, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_db, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, full_dataset.classes
