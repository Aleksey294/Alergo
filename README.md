# AIergo

AIergo — это приложение для прогнозирования аллергических реакций на основе данных о состоянии окружающей среды. Приложение анализирует введённые пользователем данные и предоставляет рекомендации, стоит ли выходить на улицу.

## Функциональность
- Ввод данных о температуре, влажности, скорости ветра и уровне пыльцы.
- Указание возраста, пола и аллергена.
- Анализ данных с помощью LLama 3.
- Отображение рекомендаций в веб-интерфейсе.


## Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://gitverse.ru/AlexAI/AIergo
cd AIergo
```
### 2. Установка необходимых библиотек и LLama 3
```bash
pip install -r requirements.txt
```
Для загрузки LLama 3 необходим Docker\
Скачать его можно тут:

https://www.docker.com

Загрузка LLama 3 на GPU
``` bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
Загрузка LLama 3 на CPU
``` bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
### 3. Запуск 
``` bash
streamlit run main.py
```