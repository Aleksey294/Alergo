from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# === ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ ===
app = FastAPI()

# Подключение HTML-шаблонов
templates = Jinja2Templates(directory="templates")

# === ОПИСАНИЕ МОДЕЛИ ===
# Упрощённый пример обученной модели
def create_model(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Создаём фиктивную модель с заранее известным количеством входов (например, 7)
model = create_model(input_dim=9)

# Загружаем обученные веса, если есть (например, model.load_weights("model.h5"))

# === СЛОВАРИ ДЛЯ КОДИРОВКИ ===
GENDER_MAP = {"male": 0, "female": 1}
ALLERGEN_MAP = {"пыльца березы": [1, 0, 0], "амброзия": [0, 1, 0], "трава": [0, 0, 1]}

# === КЛАСС ДЛЯ API ===
class UserInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    pollen_count: float
    gender: str  # "male" или "female"
    age: int
    allergen: str  # Например: "пыльца березы"

# === ГЛАВНАЯ СТРАНИЦА ===
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === API ДЛЯ ПРЕДСКАЗАНИЯ ===
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    temperature: float = Form(...),
    humidity: float = Form(...),
    wind_speed: float = Form(...),
    pollen_count: float = Form(...),
    gender: str = Form(...),
    age: int = Form(...),
    allergen: str = Form(...)
):
    # Кодировка пола и аллергена
    gender_encoded = GENDER_MAP[gender]
    allergen_encoded = ALLERGEN_MAP.get(allergen, [0, 0, 0])

    # Формируем вектор признаков
    features = np.array([
        temperature, humidity, wind_speed, pollen_count,
        gender_encoded, age
    ] + allergen_encoded).reshape(1, -1)

    # Предсказание
    prediction = model.predict(features)
    risk = "Высокий" if prediction > 0.5 else "Низкий"

    return templates.TemplateResponse("result.html", {"request": request, "risk": risk})

# === ПАПКА СТАТИЧЕСКИХ ФАЙЛОВ ===
app.mount("/static", StaticFiles(directory="static"), name="static")

# === ЗАПУСК ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
