import onnxruntime as ort
from sentence_transformers import SentenceTransformer
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor
import noisereduce as nr
from pydub import AudioSegment, silence
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import os
import soundfile as sf

# # Настройка пути к ffmpeg
# ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
# os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
# AudioSegment.ffmpeg = ffmpeg_path
# print(f"FFmpeg path set to: {ffmpeg_path}")

CATEGORIES = [
    'Покупка автомобиля',
    'Сервис и обслуживание автомобиля',
    'Кузовной ремонт',
    'Финансовые вопросы',
    'Вопрос оператору',
    'Нецелевой звонок'
]

def predict_texts(texts, onnx_path=None, device='cpu'):
    if onnx_path is None:
        onnx_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'model.onnx'))
    # print(f"Путь к модели: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Файл модели {onnx_path} не найден. Убедитесь, что он существует.")
    
    # with open(onnx_path, 'rb') as f:
    #     print("Файл model.onnx успешно открыт для чтения.")
    
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    base_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
    tokenized = base_model.tokenize(texts)
    inputs = {
        'input_ids': tokenized['input_ids'].cpu().numpy(),
        'attention_mask': tokenized['attention_mask'].cpu().numpy()
    }
    
    logits = ort_session.run(None, inputs)[0]
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(logits, axis=1)
    
    results = []
    for text, pred, prob in zip(texts, preds, probs):
        results.append({
            'text': text,
            'category': CATEGORIES[pred],
            'confidence': float(prob[pred]),
            'probabilities': {CATEGORIES[i]: float(prob[i]) for i in range(len(CATEGORIES))}
        })
    return results

def preprocess_audio(audio_path):
    # 1. Загрузка аудио
    audio, sr = librosa.load(audio_path, sr=16000)
    
# 2. Шумоподавление и усиление речевых частот
    audio_denoised = nr.reduce_noise(y=audio, sr=sr) # , prop_decrease=0.5, time_constant_s=0.5
    audio_denoised = librosa.effects.preemphasis(audio_denoised, coef=0.8)
    
    # 3. Нормализация
    audio_normalized = librosa.util.normalize(audio_denoised)
    
    # 4. Сегментация на основе тишины
    temp_wav = "temp_audio.wav"
    sf.write(temp_wav, audio_normalized, sr)
    
    audio_segment = AudioSegment.from_wav(temp_wav)
    
    segments = silence.split_on_silence(
        audio_segment,
        min_silence_len=500,
        silence_thresh=-50,
        keep_silence=200
    )
    
    combined = AudioSegment.empty()
    for segment in segments:
        combined += segment
    combined.export(temp_wav, format="wav")
    
    audio_processed, sr = librosa.load(temp_wav, sr=16000)
    
    # Сохраняем обработанный аудиофайл для воспроизведения
    processed_audio_path = "processed_audio.wav"
    sf.write(processed_audio_path, audio_processed, sr)
    
    # 5. Создание графиков волны (waveforms до и после обработки)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(audio) / sr, len(audio)), audio, color='blue')
    plt.title("Исходный аудиосигнал")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, len(audio_processed) / sr, len(audio_processed)), audio_processed, color='green')
    plt.title("Аудиосигнал после обработки")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    waveform_plots = Image.open(buf)
    
    # 6. Создание мел-спектрограмм (исходного и обработанного звука)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    S_original = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_original_dB = librosa.power_to_db(S_original, ref=np.max)
    librosa.display.specshow(S_original_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Мел-спектрограмма исходного аудио")
    plt.xlabel("Время (с)")
    plt.ylabel("Частота (мел)")
    
    plt.subplot(2, 1, 2)
    S_processed = librosa.feature.melspectrogram(y=audio_processed, sr=sr, n_mels=128)
    S_processed_dB = librosa.power_to_db(S_processed, ref=np.max)
    librosa.display.specshow(S_processed_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Мел-спектрограмма после обработки")
    plt.xlabel("Время (с)")
    plt.ylabel("Частота (мел)")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    spectrogram_plots = Image.open(buf)
    
    return audio_processed, sr, waveform_plots, spectrogram_plots, processed_audio_path

def whisper_asr_pytorch(audio_path, processor, model, max_length=1000, language="ru"):
    audio_processed, sr, waveform_plots, spectrogram_plots, processed_audio_path = preprocess_audio(audio_path)
    input_features = processor(audio_processed, sampling_rate=sr, return_tensors="pt").input_features
    
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ru", task="transcribe")
    start_tokens = processor.tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", f"<|{language}|>", "<|notimestamps|>"])
    decoder_input_ids = torch.tensor([start_tokens])
    decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)
    
    outputs = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        attention_mask=decoder_attention_mask,
        max_length=max_length
    )
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Удаляем временный файл после транскрипции
    try:
        os.remove("temp_audio.wav")
    except FileNotFoundError:
        pass
    
    return transcription, waveform_plots, spectrogram_plots, processed_audio_path