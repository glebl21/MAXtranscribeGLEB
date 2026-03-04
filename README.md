# 🎙️ MAX Voice Transcriber Bot

Бот для **мессенджера MAX**: транскрибирует голосовые/аудиофайлы в текст и делает краткое саммари.

## 🔑 Переменные окружения

- `MAX_BOT_TOKEN` — токен бота MAX
- `GROQ_API_KEY` — ключ Groq (Whisper + LLaMA)
- `GEMINI_API_KEY` — (опционально) ключ Gemini для саммари
- `MAX_API_BASE` — (опционально) базовый URL Bot API, по умолчанию `https://botapi.max.ru`
- `MAX_MESSAGE_LENGTH` — (опционально) длина одного сообщения, по умолчанию `4000`

## 🚀 Установка и запуск

```bash
pip install -r requirements.txt
export MAX_BOT_TOKEN="..."
export GROQ_API_KEY="..."
python voice_transcriber_bot.py
