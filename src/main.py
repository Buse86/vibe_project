import argparse
import os
import sys
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pdfplumber

# Поддерживаемые языки модели
SUPPORTED_LANGS = {'en', 'ru', 'de'}

# Соответствие языков для модели
LANG_MAP = {
    'en': 'english',
    'ru': 'russian',
    'de': 'german'
}

def load_text(file_path: str) -> str:
    """Загружает текст из .txt или .pdf файла"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    if file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.lower().endswith('.pdf'):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else:
        raise ValueError("Поддерживаются только .txt и .pdf файлы")

def detect_language(text: str) -> str:
    """Определяет язык текста и проверяет поддержку"""
    try:
        lang = detect(text)
    except:
        lang = 'en'  # fallback
    
    if lang not in SUPPORTED_LANGS:
        print(f"⚠️ Обнаружен неподдерживаемый язык: {lang}. Используем английский как fallback.")
        lang = 'en'
    return lang

def summarize_text(text: str, lang: str, compression: int, max_input_length=1024):
    """Абстрактивное резюмирование с помощью mT5"""
    if not text.strip():
        return "Пустой текст"
    
    # Загрузка модели и токенизатора (можно закэшировать в будущем)
    model_name = "csebuetnlp/mT5_multilingual_xlsum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Обрезаем текст до максимальной длины модели
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    # Определяем длину генерации на основе сжатия
    input_len = len(tokenizer.tokenize(text))
    target_len = max(30, int(input_len * compression / 100))
    max_len = min(512, target_len)
    min_len = max(20, int(max_len * 0.7))

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_len,
            min_length=min_len,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Multilingual Learning Material Summarizer")
    parser.add_argument("--input", "-i", required=True, help="Путь к текстовому файлу (.txt или .pdf)")
    parser.add_argument("--language", "-l", choices=["auto", "en", "ru", "de"], default="auto",
                        help="Язык текста (по умолчанию: автоопределение)")
    parser.add_argument("--compression", "-c", type=int, choices=[20, 30, 50], default=30,
                        help="Уровень сжатия в процентах (20, 30, 50)")
    parser.add_argument("--output", "-o", help="Сохранить результат в файл")

    args = parser.parse_args()

    # 1. Загрузка текста
    try:
        text = load_text(args.input)
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        sys.exit(1)

    if not text.strip():
        print("❌ Файл пустой")
        sys.exit(1)

    # 2. Определение языка
    if args.language == "auto":
        lang = detect_language(text)
    else:
        lang = args.language

    if lang not in SUPPORTED_LANGS:
        print(f"❌ Неподдерживаемый язык: {lang}")
        sys.exit(1)

    print(f"🔤 Обнаружен язык: {LANG_MAP[lang]} ({lang})")
    print(f"📉 Уровень сжатия: {args.compression}%")

    # 3. Резюмирование
    print("⏳ Генерация резюме... (может занять 1–2 минуты на CPU)")
    summary = summarize_text(text, lang, args.compression)

    # 4. Вывод
    print("\n" + "="*60)
    print("📝 РЕЗЮМЕ:")
    print("="*60)
    print(summary)

    # 5. Сохранение (опционально)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\n✅ Результат сохранён в: {args.output}")

if __name__ == "__main__":
    main()