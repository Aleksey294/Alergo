import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from openai import OpenAI
import deep_translator

# === OpenAI клиент ===
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

# === Описание модели ===
def create_model(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Создаем фиктивную модель
model = create_model(input_dim=16)  # Обновлено с учетом 10 аллергенов

# === Словари для кодировки ===
GENDER_MAP = {"male": 0, "female": 1}
ALLERGENS = [
    "пыльца березы", "амброзия", "трава", "пыльца дуба",
    "тополиный пух", "пыльца сосны", "пыльца клёна",
    "пыльца ольхи", "пыльца ясеня", "пыльца подсолнечника"
]

# Создаем кодировку аллергенов
ALLERGEN_MAP = {allergen: [1 if i == idx else 0 for i in range(len(ALLERGENS))] for idx, allergen in enumerate(ALLERGENS)}

# === Функция для предсказания ===
def predict_risk(features):
    prediction = model.predict(features)
    return "Высокий" if prediction > 0.5 else "Низкий"

# === Функция для объяснения ===
def get_explanation(user_input):
    dialog_history = [{"role": "user", "content": user_input}]
    llama_response = client.chat.completions.create(
        model="llama3:8b",
        messages=dialog_history,
    )
    return llama_response.choices[0].message.content

# === Streamlit Приложение ===
st.title("Прогноз уровня риска аллергии")

# Ввод данных
temperature = st.number_input("Температура воздуха (°C)", min_value=-50, max_value=50, value=0)
humidity = st.number_input("Влажность (%)", min_value=0, max_value=100, value=0)
wind_speed = st.number_input("Скорость ветра (м/с)", min_value=0.0, max_value=50.0, value=0.0)
pollen_count = st.number_input("Количество пыльцы", min_value=0, max_value=1000, value=0)
gender = st.selectbox("Пол", ["male", "female"])
age = st.number_input("Возраст", min_value=0, max_value=100, value=0)
selected_allergens = st.multiselect("Выберите аллергены", ALLERGENS, default=["пыльца березы"])

# Кодировка данных
gender_encoded = GENDER_MAP[gender]
allergen_encoded = np.sum([ALLERGEN_MAP[allergen] for allergen in selected_allergens], axis=0)

# Если аллергены не выбраны, массив заполняется нулями
if allergen_encoded.size == 0:
    allergen_encoded = np.zeros(len(ALLERGENS))

# Формируем вектор признаков
features = np.array([
    temperature, humidity, wind_speed, pollen_count,
    gender_encoded, age
] + list(allergen_encoded)).reshape(1, -1)

# Кнопка для прогноза
if st.button("Получить прогноз"):
    risk = predict_risk(features)

    # Генерация объяснения
    allergens_text = ", ".join(selected_allergens) if selected_allergens else "нет данных"
    user_input = (
        f"Данные о состоянии окружающей среды:\n"
        f"- Температура: {temperature}°C\n"
        f"- Влажность: {humidity}%\n"
        f"- Скорость ветра: {wind_speed} м/с\n"
        f"- Уровень пыльцы: {pollen_count}\n"
        f"Пользователь: {age} лет, пол: {gender}, аллергены: {allergens_text}.\n\n"
        f"На основании этих данных, вероятность риска: {risk}. "
        f"Объясни, почему это так, и дай рекомендации. И самое главное - пиши на РУССКОМ! И сделай текст коротким, но со смыслом."
    )
    explanation = deep_translator.GoogleTranslator(source='auto', target='ru').translate(get_explanation(user_input))

    # Вывод результатов
    st.subheader(f"Уровень риска: {risk}")
    st.subheader("Объяснение:")
    st.write(explanation)
