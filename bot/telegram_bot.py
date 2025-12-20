import os
import asyncio
import torch
import torch.nn.functional as nn_f
import joblib
import numpy as np
from PIL import Image

from aiogram import Bot, Dispatcher, types, F 
from torchvision import transforms
from dotenv import load_dotenv

from src.models import SimpleCNN, get_resnet18
from src.features import extract_hog_features

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    exit("TOKEN не найден в .env")

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

bot = Bot(token=TOKEN)
dp = Dispatcher()

device = torch.device("cpu")

# Загрузка моделей
print("Загрузка моделей...")
try:
    svm_model = joblib.load("models/svm_hog.pkl")
    
    cnn_model = SimpleCNN()
    cnn_model.load_state_dict(torch.load("models/simple_cnn.pth", map_location=device))
    cnn_model.eval()

    resnet_model = get_resnet18()
    resnet_model.load_state_dict(torch.load("models/resnet18.pth", map_location=device))
    resnet_model.eval()
    print("Все модели успешно загружены!")
except FileNotFoundError as e:
    exit(f"Ошибка: Не найден файл модели. Сначала запустите обучение (train_all.py). {e}")


# Препроцессинг
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_top2_prediction(probs_tensor):
    """Возвращает списки вероятностей и индексов для топ-2 классов."""
    top2_prob, top2_idx = torch.topk(probs_tensor, 2)
    return top2_prob.tolist(), top2_idx.tolist()

@dp.message(F.text == "/start")
async def start(message: types.Message):
    welcome_text = (
        "🛰️ **EuroSAT Classifier Bot**\n\n"
        "Я определяю тип местности на спутниковых снимках. \n"
        "Я показываю **Топ-2** вероятных класса для каждой модели.\n\n"
        "Просто отправь мне картинку!"
    )
    await message.answer(welcome_text, parse_mode="Markdown")

@dp.message(F.photo)
async def classify(message: types.Message):
    status_msg = await message.answer("Satelite processing... 🛰️")
    
    # Скачивание и подготовка изображения
    file = await bot.get_file(message.photo[-1].file_id)
    photo_bytes = await bot.download_file(file.file_path)
    img = Image.open(photo_bytes).convert("RGB")

    # Тензор для нейросетей
    input_tensor = preprocess(img).unsqueeze(0)
    
    # Признаки для SVM
    feat_hog = extract_hog_features(input_tensor[0])

    with torch.no_grad():
        # 1 ResNet18
        res_logits = resnet_model(input_tensor)

        res_probs = nn_f.softmax(res_logits, dim=1)[0]
        res_top2_p, res_top2_i = get_top2_prediction(res_probs)

        # 2 Simple CNN
        cnn_logits = cnn_model(input_tensor)

        cnn_probs = nn_f.softmax(cnn_logits, dim=1)[0]
        cnn_top2_p, cnn_top2_i = get_top2_prediction(cnn_probs)

    # 3 SVM
    svm_probs_np = svm_model.predict_proba(feat_hog)[0]
    svm_top2_i = svm_probs_np.argsort()[-2:][::-1]
    svm_top2_p = svm_probs_np[svm_top2_i]
    
    def format_line(model_name, indices, probs):
        c1, c2 = CLASSES[indices[0]], CLASSES[indices[1]]
        p1, p2 = probs[0] * 100, probs[1] * 100
        return f"*{model_name}*:\n  🥇 {c1} ({p1:.1f}%)\n  🥈 {c2} ({p2:.1f}%)"

    resp = (
        "📊 **Результаты классификации (Top-2):**\n\n"
        f"{format_line('ResNet18 🏆', res_top2_i, res_top2_p)}\n\n"
        f"{format_line('Simple CNN', cnn_top2_i, cnn_top2_p)}\n\n"
        f"{format_line('SVM + HOG', svm_top2_i, svm_top2_p)}"
    )

    await status_msg.delete()
    await message.answer(resp, parse_mode="Markdown")

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nБот остановлен пользователем (Ctrl+C).")