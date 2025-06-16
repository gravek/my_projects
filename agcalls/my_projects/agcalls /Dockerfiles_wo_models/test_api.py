import pytest
from api_app import predict_texts
import os

model_onnx_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "model.onnx"))

# Тесты для каждой категории
def test_predict_texts_purchase():
    result = predict_texts(["Я хочу купить новую машину"], onnx_path=model_onnx_path)[0]
    assert result["category"] == "Покупка автомобиля"

def test_predict_texts_service():
    result = predict_texts(["Мне нужно записаться на ТО"], onnx_path=model_onnx_path)[0]
    assert result["category"] == "Сервис и обслуживание автомобиля"

def test_predict_texts_repair():
    result = predict_texts(["Нужно починить кузов после аварии"], onnx_path=model_onnx_path)[0]
    assert result["category"] == "Кузовной ремонт"

def test_predict_texts_finance():
    result = predict_texts(["Какие у вас условия автокредита?"], onnx_path=model_onnx_path)[0]
    assert result["category"] == "Финансовые вопросы"

def test_predict_texts_operator():
    result = predict_texts(["Соедините меня с оператором"], onnx_path=model_onnx_path)[0]
    assert result["category"] == "Вопрос оператору"

def test_predict_texts_non_target():
    result = predict_texts(["Продам картошку оптом"], onnx_path=model_onnx_path)[0]
    assert result["category"] == "Нецелевой звонок"

if __name__ == "__main__":
    pytest.main()