import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.utils import load_text, detect_language, summarize_text
from src.main import main  # предполагается, что main() принимает аргументы через argparse


# Тестовые тексты
SAMPLE_EN = "Artificial intelligence is a wonderful field. It enables machines to learn from data."
SAMPLE_RU = "Искусственный интеллект — это замечательная область. Он позволяет машинам обучаться на данных."
SAMPLE_DE = "Künstliche Intelligenz ist ein wunderbares Gebiet. Sie ermöglicht Maschinen, aus Daten zu lernen."


def test_load_text_txt():
    """Тест загрузки .txt файла"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(SAMPLE_RU)
        temp_path = f.name

    try:
        text = load_text(temp_path)
        assert SAMPLE_RU in text
    finally:
        os.unlink(temp_path)


def test_load_text_pdf():
    """Тест загрузки .pdf файла (требуется pdfplumber, но не реальный PDF)"""
    # Пропускаем тест, если нет реального PDF или в CI
    pytest.skip("PDF parsing test skipped in CI (requires real text-based PDF)")


def test_detect_language():
    """Тест определения языка"""
    assert detect_language(SAMPLE_EN) == 'en'
    assert detect_language(SAMPLE_RU) == 'ru'
    assert detect_language(SAMPLE_DE) == 'de'


@patch('src.utils.AutoTokenizer')
@patch('src.utils.AutoModelForSeq2SeqLM')
@patch('src.utils.torch')
def test_summarize_text(mock_torch, mock_model, mock_tokenizer):
    """Тест функции summarize_text с моками (без загрузки реальной модели)"""
    # Настройка моков
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.tokenize.return_value = ['token'] * 100
    mock_tokenizer_instance.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}
    mock_tokenizer_instance.decode.return_value = "Сгенерированное резюме."

    mock_model_instance = MagicMock()
    mock_model.from_pretrained.return_value = mock_model_instance
    mock_model_instance.generate.return_value = [MagicMock()]

    # Вызов функции
    result = summarize_text(SAMPLE_RU, lang='ru', compression=30)

    # Проверка
    assert result == "Сгенерированное резюме."
    mock_model.from_pretrained.assert_called_once()
    mock_tokenizer.from_pretrained.assert_called_once()


@patch('sys.argv', ['main', '-i', 'data/test_en.txt', '-c', '30'])
@patch('src.main.summarize_text')
@patch('src.main.load_text')
def test_main_integration(mock_load, mock_summarize, capsys):
    """Тест основного CLI-потока (интеграционный, но без модели)"""
    mock_load.return_value = SAMPLE_EN
    mock_summarize.return_value = "AI is a wonderful field."

    # Запуск основной функции
    from src.main import main
    main()

    # Проверка вывода
    captured = capsys.readouterr()
    assert "РЕЗЮМЕ" in captured.out
    assert "AI is a wonderful field." in captured.out