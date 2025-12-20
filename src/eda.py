import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Exploratory Data Analysis

def analyze_dataset(data_path: str):
    """
    Полная EDA набора данных с изображениями:
    - Балансировка классов
    - Размеры и форматы изображений
    - Проверка пропущенных и повреждённых файлов
    - Статистика размеров и выбросы
    """
    stats = []
    missing_files = []
    sizes = []

    # Получаем список классов (папок)
    classes = [d for d in os.listdir(data_path)
               if os.path.isdir(os.path.join(data_path, d)) and d != 'allBands']

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not files:
            continue

        for f in files:
            file_path = os.path.join(cls_path, f)
            try:
                img = Image.open(file_path)
                img.verify()  # проверка на повреждённость
                img = Image.open(file_path)  # повторное открытие для анализа размеров
                sizes.append((img.size[0], img.size[1]))
            except Exception:
                missing_files.append(file_path)

        # Берем первый файл для примера формата и размера
        sample_img = Image.open(os.path.join(cls_path, files[0]))
        stats.append({
            'Класс': cls,
            'Кол-во': len(files),
            'Пример размера': f"{sample_img.size[0]}x{sample_img.size[1]}",
            'Формат': sample_img.format
        })

    # Таблица классов
    df = pd.DataFrame(stats)
    print("Балансировка классов:\n", df.to_string(index=False))

    if missing_files:
        print("\nПовреждённые или недоступные файлы:")
        for f in missing_files:
            print(f)

    # Статистика размеров
    sizes_df = pd.DataFrame(sizes, columns=["width", "height"])
    print("\nСтатистика размеров изображений:\n", sizes_df.describe())

    # График распределения классов
    plt.figure(figsize=(10, 5))
    plt.bar(df['Класс'], df['Кол-во'], color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Распределение классов")
    plt.tight_layout()
    plt.savefig("eda_class_distribution.png")
    plt.close()

    # Гистограмма размеров
    plt.figure(figsize=(10, 5))
    plt.hist(sizes_df["width"], bins=20, alpha=0.5, label="Width")
    plt.hist(sizes_df["height"], bins=20, alpha=0.5, label="Height")
    plt.title("Распределение ширины и высоты изображений")
    plt.xlabel("Пиксели")
    plt.ylabel("Количество")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda_image_sizes.png")
    plt.close()

    print("\nГрафики сохранены: eda_class_distribution.png, eda_image_sizes.png")
    return df, sizes_df, missing_files

if __name__ == "__main__":
    data_path = "./data/"
    
    df = analyze_dataset(data_path)