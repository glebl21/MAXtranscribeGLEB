import hashlib
import logging
import os
import tempfile
import time
from collections import defaultdict
from typing import Any

import requests
from groq import Groq

MAX_BOT_TOKEN = os.environ.get("MAX_BOT_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_API_BASE = os.environ.get("MAX_API_BASE", "https://botapi.max.ru")
MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "4000"))
MAX_FILE_SIZE_BYTES = 19 * 1024 * 1024

if not MAX_BOT_TOKEN:
    raise RuntimeError("Environment variable MAX_BOT_TOKEN is required")
if not GROQ_API_KEY:
    raise RuntimeError("Environment variable GROQ_API_KEY is required")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_API_KEY)

transcription_store: dict[str, str] = {}
stats = defaultdict(lambda: {"count": 0, "summaries": 0})
last_text_by_chat: dict[str, str] = {}


class MaxApiClient:
    def __init__(self, token: str, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("ok") is False:
            raise RuntimeError(payload.get("description") or f"MAX API error: {payload}")
        return payload

    def get_updates(self, offset: int | None = None, timeout: int = 25) -> list[dict[str, Any]]:
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        data = self._request("GET", "/updates", params=params)
        if isinstance(data, dict):
            if "result" in data and isinstance(data["result"], list):
                return data["result"]
            if "updates" in data and isinstance(data["updates"], list):
                return data["updates"]
        return []

    def send_message(self, chat_id: str, text: str) -> None:
        self._request("POST", "/messages", json={"chat_id": chat_id, "text": text})

    def download_file(self, file_id: str) -> tuple[bytes, str, int | None]:
        file_info = self._request("GET", f"/files/{file_id}")
        result = file_info.get("result", file_info)
        file_url = result.get("url") or result.get("download_url")
        if not file_url:
            raise RuntimeError("MAX API did not return file download URL")

        filename = result.get("name") or result.get("file_name") or "audio.ogg"
        size = result.get("size")
        if size and size > MAX_FILE_SIZE_BYTES:
            raise RuntimeError("Файл слишком большой. Допустимый размер до 20 МБ.")

        response = requests.get(file_url, timeout=60)
        response.raise_for_status()
        return response.content, filename, size


max_client = MaxApiClient(MAX_BOT_TOKEN, MAX_API_BASE)


def store_text(text: str) -> str:
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    transcription_store[key] = text
    return key


def split_for_max(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    chunks = []
    remaining = text.strip()

    while len(remaining) > max_len:
        split_at = remaining.rfind("\n", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    ext = os.path.splitext(filename)[1] or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3",
                language=None,
                response_format="text",
            )
        return transcription.strip()
    finally:
        os.unlink(tmp_path)


def summarize_text(text: str) -> dict[str, str]:
    if GEMINI_API_KEY:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
            )
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    "Сделай краткое саммари этого текста. "
                                    "Выдели ключевые мысли и выводы. "
                                    "Отвечай на том же языке что и текст. "
                                    "Используй маркированный список (•).\n\n" + text
                                )
                            }
                        ]
                    }
                ]
            }
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                return {"text": result, "model": "Gemini 2.0 Flash ✨"}
        except Exception:
            logger.exception("Gemini summarize failed, fallback to Groq")

    result = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Сделай краткое саммари текста. "
                    "Выдели ключевые мысли и выводы. "
                    "Отвечай на том же языке что и текст. "
                    "Используй маркированный список (•)."
                ),
            },
            {"role": "user", "content": text},
        ],
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        temperature=0.4,
    )
    return {"text": result.choices[0].message.content.strip(), "model": "Groq LLaMA 3.3 70B ⚡"}


def send_chunked(chat_id: str, text: str) -> None:
    for chunk in split_for_max(text):
        max_client.send_message(chat_id, chunk)


def process_audio(chat_id: str, user_id: str, file_id: str, fallback_filename: str = "audio.ogg") -> None:
    send_chunked(chat_id, "⏳ Транскрибирую...")

    try:
        audio_bytes, filename, _ = max_client.download_file(file_id)
        final_filename = filename or fallback_filename
        text = transcribe_audio(audio_bytes, final_filename)
        if not text:
            send_chunked(chat_id, "❌ Не удалось распознать речь.")
            return

        stats[user_id]["count"] += 1
        text_key = store_text(text)
        last_text_by_chat[chat_id] = text_key

        send_chunked(chat_id, f"📄 Транскрипция:\n\n{text}")
        send_chunked(chat_id, "Для саммари отправь команду: /summary")
    except Exception as exc:
        send_chunked(chat_id, f"❌ Ошибка: {exc}")
        logger.exception("Failed to process audio")


def handle_text_command(chat_id: str, user_id: str, text: str) -> None:
    normalized = text.strip().lower()

    if normalized in {"/start", "/help"}:
        send_chunked(
            chat_id,
            "🎙️ Бот-транскрибатор для MAX\n\n"
            "Отправь голосовое или аудиофайл — переведу в текст.\n"
            "После транскрибации используй /summary для краткого изложения.\n"
            "Команда /stats покажет твою статистику.",
        )
        return

    if normalized == "/stats":
        send_chunked(
            chat_id,
            f"📊 Твоя статистика:\n\n"
            f"• Транскрибировано: {stats[user_id]['count']}\n"
            f"• Саммари сделано: {stats[user_id]['summaries']}",
        )
        return

    if normalized == "/summary":
        text_key = last_text_by_chat.get(chat_id)
        if not text_key:
            send_chunked(chat_id, "❌ Сначала отправь голосовое сообщение или аудиофайл.")
            return

        source_text = transcription_store.get(text_key)
        if not source_text:
            send_chunked(chat_id, "❌ Текст не найден. Отправь аудио заново.")
            return

        try:
            send_chunked(chat_id, "⏳ Генерирую саммари...")
            summary = summarize_text(source_text)
            stats[user_id]["summaries"] += 1
            send_chunked(chat_id, f"📝 Краткое изложение (от {summary['model']}):\n\n{summary['text']}")
        except Exception as exc:
            send_chunked(chat_id, f"❌ Ошибка саммари: {exc}")
            logger.exception("Failed to summarize")


def extract_payload(update: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    message = update.get("message") or update.get("result") or update
    chat = message.get("chat") or {}
    sender = message.get("from") or message.get("sender") or {}

    chat_id = str(chat.get("id") or message.get("chat_id") or "") or None
    user_id = str(sender.get("id") or message.get("user_id") or chat_id or "") or None

    text = message.get("text")

    attachment = message.get("audio") or message.get("voice") or message.get("document") or message.get("file")
    file_id = None
    filename = None
    if isinstance(attachment, dict):
        file_id = attachment.get("file_id") or attachment.get("id")
        filename = attachment.get("file_name") or attachment.get("name")

    return chat_id, user_id, text, file_id, filename


def run_polling() -> None:
    logger.info("🤖 MAX бот запущен")
    offset = None

    while True:
        try:
            updates = max_client.get_updates(offset=offset, timeout=25)
            for update in updates:
                update_id = update.get("update_id") or update.get("id")
                if isinstance(update_id, int):
                    offset = update_id + 1

                chat_id, user_id, text, file_id, filename = extract_payload(update)
                if not chat_id or not user_id:
                    continue

                if text:
                    handle_text_command(chat_id, user_id, text)
                    continue

                if file_id:
                    process_audio(chat_id, user_id, file_id, filename or "audio.ogg")
        except requests.RequestException:
            logger.exception("MAX API request failed")
            time.sleep(3)
        except Exception:
            logger.exception("Unexpected polling error")
            time.sleep(3)


if __name__ == "__main__":
    run_polling()
