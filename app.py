import threading
import queue
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Глобальная очередь для запросов
request_queue = queue.Queue()
# Словарь для хранения результатов (ключ — request_id)
results = {}
# Блокировка для безопасного доступа к results
results_lock = threading.Lock()

# Глобальные переменные для модели
model = None
tokenizer = None
token = os.getenv('TOKEN')

def load_model():
    """Загрузка модели один раз при старте"""
    global model, tokenizer
    print("Загрузка модели...")
    model = SentenceTransformer("ai-forever/FRIDA", token=token)
    print("Модель загружена!")


def worker():
    """Поток-обработчик запросов из очереди"""
    while True:
        # Получаем запрос из очереди
        request_id, prompt, event = request_queue.get()
        if prompt is None:  # Сигнал остановки
            break

        try:
            # Генерация ответа
            embedding = model.encode(prompt, prompt_name="search_document")

            # Сохраняем результат с блокировкой
            with results_lock:
                results[request_id] = {
                    'status': 'completed',
                    'response': embedding
                }
        except Exception as e:
            with results_lock:
                results[request_id] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Сигнализируем, что обработка завершена
        event.set()
        request_queue.task_done()

@app.route('/ask', methods=['POST'])
def ask():
    """API endpoint для отправки запросов и получения ответов"""
    data = request.json
    prompt = data.get('prompt', '')
    request_id = data.get('request_id', f'req-{int(time.time() * 1000)}')

    # Создаём событие для синхронизации
    event = threading.Event()

    # Добавляем запрос в очередь
    request_queue.put((request_id, prompt, event))

    # Ждём завершения обработки (с таймаутом)
    if event.wait(timeout=30):  # Таймаут 30 секунд
        # Получаем результат с блокировкой
        with results_lock:
            result = results.get(request_id, {})
            # Удаляем обработанный запрос из кэша (опционально)
            if request_id in results:
                del results[request_id]

        if result.get('status') == 'completed':
            return jsonify({
                'status': 'success',
                'request_id': request_id,
                'response': str(result['response'])
            })
        else:
            return jsonify({
                'status': 'error',
                'request_id': request_id,
                'error': result.get('error', 'Unknown error')
            })
    else:
        return jsonify({
            'status': 'timeout',
            'request_id': request_id,
            'error': 'Превышено время ожидания ответа от модели'
        }), 504

if __name__ == '__main__':
    # Загружаем модель
    load_model()
    # Запускаем рабочий поток
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    # Запускаем Flask-сервер
    app.run(host='0.0.0.0', port=5000)
