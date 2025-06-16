import gradio as gr
from api_app import predict_texts, whisper_asr_pytorch, CATEGORIES
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

# Проверка наличия model.onnx
model_onnx_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "model.onnx"))
print(f"Проверка пути model.onnx: {model_onnx_path}")
if not os.path.exists(model_onnx_path):
    raise FileNotFoundError(f"Файл {model_onnx_path} не найден")

# Проверка кэша модели Whisper
hf_cache_path = os.getenv("HF_HOME", "/root/.cache/huggingface")
whisper_model_path = os.path.join(hf_cache_path, "hub", "models--openai--whisper-small")
print(f"Проверка пути к модели Whisper: {whisper_model_path}")
if os.path.exists(hf_cache_path):
    print(f"Содержимое HF_HOME: {os.listdir(hf_cache_path)}")
if os.path.exists(os.path.join(hf_cache_path, "hub")):
    print(f"Содержимое hub: {os.listdir(os.path.join(hf_cache_path, 'hub'))}")
if os.path.exists(whisper_model_path):
    print(f"Содержимое models--openai--whisper-small: {os.listdir(whisper_model_path)}")
else:
    raise FileNotFoundError(f"Модель Whisper не найдена в {whisper_model_path}")

# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

processor = WhisperProcessor.from_pretrained(whisper_model_path)
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path)




def format_probabilities(probabilities_dict):
    # Форматируем вероятности в виде таблицы с улучшенным выравниванием (без Markdown, с процентами)
    table = "Вероятности категорий:\n\n"
    # table += "Категория                          Вероятность (%)\n"
    # table += "-----------------------------------------------\n"
    for category, prob in probabilities_dict.items():
        table += f"{category} -- {prob * 100:.2f}%\n"
        # table += f"{category:<35[:42]}{prob * 100:>10.2f}%\n"
    return table



def process_text(text_input):
    if not text_input:
        return "Пожалуйста, введите текст.", None, None, None
    
    try:
        results = predict_texts([text_input], onnx_path=model_onnx_path)
        result = results[0]
        probabilities_table = format_probabilities(result['probabilities'])
        output_text = (
            f"Текст:\t{result['text']}\n\n"
            f"Категория:\t{result['category']}\n\n"
            f"Уверенность:\t{result['confidence'] * 100:.2f}%\n\n\n"
            f"{probabilities_table}"
        )
        return output_text, None, None, None
    except Exception as e:
        return f"Ошибка при обработке текста: {str(e)}", None, None, None

def process_audio(audio_input):
    if audio_input is None:
        return "Пожалуйста, загрузите аудиофайл.", None, None, None
    
    try:
        audio_path = audio_input
        transcription, waveform_plots, spectrogram_plots, processed_audio_path = whisper_asr_pytorch(audio_path, processor, model, language="ru")
        
        classification_result = predict_texts([transcription], onnx_path=model_onnx_path)[0]
        probabilities_table = format_probabilities(classification_result['probabilities'])
        output_text = (
            f"Транскрипция: {transcription}\n\n"
            f"Классификация:\n"
            f"Категория: {classification_result['category']}\n"
            f"Уверенность: {classification_result['confidence'] * 100:.2f}%\n\n"
            f"{probabilities_table}"
        )
        return output_text, processed_audio_path, waveform_plots, spectrogram_plots
    except Exception as e:
        # Удаляем временные файлы в случае ошибки
        for temp_file in ["temp_audio.wav", "processed_audio.wav"]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        return f"Ошибка при обработке аудио: {str(e)}", None, None, None

with gr.Blocks() as demo:
    gr.Markdown("# Демонстрация классификации текстов и транскрипции аудио")
    
    with gr.Tab("Классификация текста"):
        text_input = gr.Textbox(label="Введите текст на русском языке")
        text_output = gr.Textbox(label="Результат")
        text_button = gr.Button("Классифицировать текст")
        text_button.click(
            process_text,
            inputs=[text_input],
            outputs=[
                text_output,
                gr.Audio(label="Обработанный звук", visible=False),
                gr.Image(label="Графики волны", visible=False),
                gr.Image(label="Мел-спектрограммы", visible=False)
            ]
        )
    
    with gr.Tab("Транскрипция и классификация аудио"):
        audio_input = gr.Audio(type="filepath", label="Загрузите аудиофайл (русский язык)")
        audio_button = gr.Button("Транскрибировать и классифицировать")  # Кнопка сразу после загрузки
        audio_output = gr.Textbox(label="Результат")
        processed_audio_output = gr.Audio(label="Обработанный звук")  # Воспроизведение обработанного звука
        waveform_plots_output = gr.Image(label="Графики волны")
        spectrogram_plots_output = gr.Image(label="Мел-спектрограммы")
        audio_button.click(
            process_audio,
            inputs=[audio_input],
            outputs=[
                audio_output,
                processed_audio_output,
                waveform_plots_output,
                spectrogram_plots_output
            ]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    # Удаляем временные файлы после завершения работы
    for temp_file in ["temp_audio.wav", "processed_audio.wav"]:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass