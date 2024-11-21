import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

# === ОПИСАНИЕ МОДЕЛИ ===
def create_model(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Создаём фиктивную модель
model = create_model(input_dim=9)

# === СЛОВАРИ ДЛЯ КОДИРОВКИ ===
GENDER_MAP = {"male": 0, "female": 1}
ALLERGEN_MAP = {"пыльца березы": [1, 0, 0], "амброзия": [0, 1, 0], "трава": [0, 0, 1]}

# === ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ===
def predict_risk(features):
    prediction = model.predict(features)
    return "Высокий" if prediction > 0.5 else "Низкий"

# === ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ ОБЪЯСНЕНИЯ ===
def get_explanation(user_input):
    dialog_history = [{"role": "user", "content": user_input}]
    llama_response = client.chat.completions.create(
        model="llama3:8b",
        messages=dialog_history,
    )
    return llama_response.choices[0].message.content

# === Streamlit Приложение ===
st.title("Прогноз уровня риска аллергии")

# Формы для ввода данных
temperature = st.number_input("Температура воздуха (°C)", min_value=-50, max_value=50, value=25)
humidity = st.number_input("Влажность (%)", min_value=0, max_value=100, value=60)
wind_speed = st.number_input("Скорость ветра (м/с)", min_value=0.0, max_value=50.0, value=5.0)
pollen_count = st.number_input("Количество пыльцы", min_value=0, max_value=1000, value=200)
gender = st.selectbox("Пол", ["Мужчина", "Женщина"])
age = st.number_input("Возраст", min_value=0, max_value=100, value=30)
allergen = st.selectbox("Аллерген", ["пыльца березы", "амброзия", "трава"])

# Кодируем входные данные
gender_encoded = GENDER_MAP[gender]
allergen_encoded = ALLERGEN_MAP.get(allergen, [0, 0, 0])

# Формируем вектор признаков
features = np.array([
    temperature, humidity, wind_speed, pollen_count,
    gender_encoded, age
] + allergen_encoded).reshape(1, -1)

# Прогнозирование риска
if st.button("Получить прогноз"):
    risk = predict_risk(features)

    # Генерация объяснения
    user_input = f"Данные о состоянии окружающей среды:\n"\
        f"- Температура: {temperature}°C\n"\
        f"- Влажность: {humidity}%\n"\
        f"- Скорость ветра: {wind_speed} м/с\n"\
        f"- Уровень пыльцы: {pollen_count}\n"\
        f"Пользователь: {age} лет, пол: {gender}, аллерген: {allergen}.\n\n"\
        f"На основании этих данных, вероятность риска: {risk}. "\
        f"Объясни, почему это так, и дай рекомендации. И самое главное - пиши на РУССКОМ! И сделай текст коротким, но со смыслом"  # Можно подставить информацию из формы для более точного объяснения
    explanation = get_explanation(user_input)

    # Выводим результаты
    st.subheader(f"Уровень риска: {risk}")
    st.subheader("Объяснение:")
    st.write(explanation)
