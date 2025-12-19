import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def analyze_dataset(data_path: str):
    """
    Исследует структуру папки с данными:
     - баланс классов
     - размеры, форматы
     - пропуски
    """
    stats = []
    classes = [d for d in os.listdir(data_path)
               if os.path.isdir(os.path.join(data_path, d)) and d != 'allBands']

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not files:
            continue

        sample_img = Image.open(os.path.join(cls_path, files[0]))
        stats.append({
            'Класс': cls,
            'Кол-во': len(files),
            'Размер': f"{sample_img.size[0]}x{sample_img.size[1]}",
            'Формат': sample_img.format
        })

    df = pd.DataFrame(stats)
    print("Балансировка классов:\n", df.to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.bar(df['Класс'], df['Кол-во'], color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Распределение классов")
    plt.tight_layout()
    plt.savefig("eda_distribution.png")
    plt.close()

    return df
