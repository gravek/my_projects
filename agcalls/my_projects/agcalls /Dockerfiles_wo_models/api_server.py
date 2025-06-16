import logging
import os
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Depends
from fastapi.security import APIKeyHeader
from celery import Celery
from api_app import predict_texts, whisper_asr_pytorch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI с описанием приложения
app = FastAPI(
    title="Демо API колл-центра",
    description="""API для транскрипции и классификации аудиозвонков (Продажа автомобиля, Покупка автомобиля, Сервис и обслуживание автомобиля, Кузовной ремонт, Финансовые вопросы, Вопрос оператору, Нецелевой звонок). Поддерживает асинхронную обработку аудиофайлов, классификацию текста и предоставляет интерактивный интерфейс через Gradio.""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Инициализация Whisper процессора и модели
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
hf_cache_path = os.getenv("HF_HOME", "/root/.cache/huggingface")
whisper_model_path = os.path.join(hf_cache_path, "hub", "models--openai--whisper-small")
processor = WhisperProcessor.from_pretrained(whisper_model_path)
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path)

model_onnx_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "model.onnx"))
logger.info(f"Путь к модели: {model_onnx_path}")


# Инициализация Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=50,
    broker_connection_retry_delay=2.0
)


# # Настройка Celery для асинхронной обработки
# celery_app = Celery('tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')



# Аутентификация
API_KEY = "demo_key_123"
API_KEY_USAGE_LIMIT = 100
api_key_usage = 0
api_key_header = APIKeyHeader(name="X-API-Key", description="API-ключ для аутентификации. Используйте `demo_key_123`. Лимит: 100 запросов.")

def verify_api_key(api_key: str = Depends(api_key_header)):
    global api_key_usage
    if api_key != API_KEY:
        logger.warning(f"Неверный API-ключ: {api_key}")
        raise HTTPException(status_code=403, detail="Неверный API-ключ")
    if api_key_usage >= API_KEY_USAGE_LIMIT:
        logger.error("Превышен лимит использования API-ключа")
        raise HTTPException(status_code=429, detail="Превышен лимит использования API-ключа")
    api_key_usage += 1
    logger.info(f"API-ключ использован {api_key_usage} раз")
    return api_key

# Микросервис 1: Транскрипция (асинхронная задача)
@celery_app.task
def transcribe_audio(audio_path: str):
    logger.info(f"Начало транскрипции файла: {audio_path}")
    try:
        transcription, waveform_plots, spectrogram_plots, processed_audio_path = whisper_asr_pytorch(
            audio_path, processor, model, language="ru"
        )
        logger.info(f"Транскрипция завершена: {transcription}")
        return transcription, processed_audio_path
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {str(e)}")
        raise e

# Микросервис 2: Классификация
def classify_text(transcription: str):
    logger.info(f"Начало классификации текста: {transcription}")
    try:
        classification_result = predict_texts([transcription], onnx_path=model_onnx_path)[0]
        probabilities = {cat: f"{prob * 100:.2f}%" for cat, prob in classification_result['probabilities'].items()}
        result = {
            "category": classification_result['category'],
            "confidence": f"{classification_result['confidence'] * 100:.2f}%",
            "probabilities": probabilities
        }
        logger.info(f"Классификация завершена: {result['category']}")
        return result
    except Exception as e:
        logger.error(f"Ошибка классификации: {str(e)}")
        raise e

@app.post(
    "/classify-call/",
    summary="Классификация аудиозвонка",
    description="""
    Эндпоинт для транскрипции и классификации аудиозвонка. Принимает WAV-файл, выполняет транскрипцию и классифицирует текст в одну из категорий.

    **Ограничения**:
    - Формат файла: только WAV.
    - Максимальная длительность: 30 секунд.
    - Требуется API-ключ (`X-API-Key`).

    **Аутентификация**:
    - API-ключ: `demo_key_123`.
    - Лимит использования: 100 запросов.

    **Пример запроса (curl)**:
    ```bash
    curl -X POST "http://localhost:8000/classify-call/?call_id=12345" \\
         -H "X-API-Key: demo_key_123" \\
         -F "audio_file=@path_to_audio.wav"
    ```
    """
)
async def classify_call(
    call_id: str = Query(..., description="Идентификатор звонка, например, `12345`"),
    audio_file: UploadFile = File(..., description="Аудиофайл в формате WAV, не более 30 секунд"),
    api_key: str = Depends(verify_api_key)
):
    logger.info(f"Получен запрос на классификацию звонка: call_id={call_id}")

    # Проверка формата файла
    if not audio_file.filename.endswith(".wav"):
        logger.warning(f"Неподдерживаемый формат файла: {audio_file.filename}")
        raise HTTPException(status_code=400, detail="Только WAV-файлы разрешены")

    # Проверка длительности файла (примерно 30 секунд)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_audio_path = temp_file.name
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
    
    audio = AudioSegment.from_wav(temp_audio_path)
    duration_ms = len(audio)
    if duration_ms > 30000:  # 30 секунд
        os.remove(temp_audio_path)
        logger.warning(f"Файл слишком длинный: {duration_ms} мс")
        raise HTTPException(status_code=400, detail="Файл не должен превышать 30 секунд")

    # Запуск асинхронной задачи транскрипции
    logger.info(f"Отправка задачи транскрипции для файла: {temp_audio_path}")
    task = transcribe_audio.delay(temp_audio_path)
    try:
        transcription, processed_audio_path = task.get(timeout=300)  # Ожидание до 5 минут
    except Exception as e:
        os.remove(temp_audio_path)
        logger.error(f"Ошибка выполнения задачи транскрипции: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка транскрипции: {str(e)}")

    # Классификация
    try:
        classification = classify_text(transcription)
    except Exception as e:
        os.remove(temp_audio_path)
        os.remove(processed_audio_path)
        logger.error(f"Ошибка классификации: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {str(e)}")

    # Формирование ответа
    result = {
        "call_id": call_id,
        "transcription": transcription,
        "category": classification["category"],
        "confidence": classification["confidence"],
        "probabilities": classification["probabilities"]
    }

    # Удаление временных файлов
    for temp_file in [temp_audio_path, processed_audio_path, "temp_audio.wav"]:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Ошибка удаления файла {temp_file}: {str(e)}")

    logger.info(f"Запрос успешно обработан: call_id={call_id}, category={classification['category']}")
    return result