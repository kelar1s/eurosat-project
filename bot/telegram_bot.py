import os
import asyncio
import torch
import torch.nn.functional as F
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
svm_model = joblib.load("models/svm_hog.pkl")
cnn_model = SimpleCNN()
cnn_model.load_state_dict(torch.load("models/simple_cnn.pth", map_location=device))
cnn_model.eval()

resnet_model = get_resnet18()
resnet_model.load_state_dict(torch.load("models/resnet18.pth", map_location=device))
resnet_model.eval()

# Препроцессинг
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@dp.message(F.text == "/start")
async def start(message: types.Message):
    welcome_text = (
        "EuroSAT Classifier Bot!\n\n"
        "Могу классифицировать спутниковые снимки на 10 категорий:\n\n"
        "1. AnnualCrop - Однолетние культуры\n"
        "2. Forest - Лес\n"
        "3. HerbaceousVegetation - Травянистая растительность\n"
        "4. Highway - Шоссе\n"
        "5. Industrial - Промышленная зона\n"
        "6. Pasture - Пастбище\n"
        "7. PermanentCrop - Многолетние культуры\n"
        "8. Residential - Жилая зона\n"
        "9. River - Река\n"
        "10. SeaLake - Море/Озеро\n\n"
        "Просто отправьте мне спутниковый снимок, и я определю его класс!\n\n"
        "Доступные модели: CNN, HOG+SVM, ResNet18"
    )
    await message.answer(welcome_text)

@dp.message(F.photo)
async def classify(message: types.Message):
    status = await message.answer("Обрабатываю...")
    file = await bot.get_file(message.photo[-1].file_id)
    photo_bytes = await bot.download_file(file.file_path)
    img = Image.open(photo_bytes).convert("RGB")

    input_tensor = preprocess(img).unsqueeze(0)
    feat_hog = extract_hog_features(input_tensor[0])

    with torch.no_grad():
        res_out = resnet_model(input_tensor)
        probs = torch.nn.functional.softmax(res_out[0], dim=0)

        cnn_out = cnn_model(input_tensor).argmax(1).item()

    svm_idx = svm_model.predict(feat_hog)[0]
    res_idx = probs.argmax().item()

    resp = (
        f"📊 **Результаты:**\n"
        f"• SVM+HOG: {CLASSES[svm_idx]}\n"
        f"• Simple CNN: {CLASSES[cnn_out]}\n"
        f"• ResNet18: {CLASSES[res_idx]} ({probs[res_idx]:.1%})"
    )
    await status.delete()
    await message.answer(resp, parse_mode="Markdown")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
